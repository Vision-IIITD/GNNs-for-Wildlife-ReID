import os
import sys
from typing import Tuple, List

sys.path.append(os.path.join(os.path.dirname(__file__), "LightGlue"))
sys.path.append(os.path.join(os.path.dirname(__file__), "SuperGluePretrainedNetwork"))
sys.path.append(os.path.join(os.path.dirname(__file__), "omniglue"))

import numpy as np
import pandas as pd
import torch
import timm
import torchvision.transforms as T
from argparse import ArgumentParser
from tqdm import tqdm
from PIL import Image

from wildlife_datasets import analysis, datasets, loader, splits
from wildlife_tools.features import DeepFeatures
from wildlife_tools.data import WildlifeDataset 
from wildlife_tools.similarity import CosineSimilarity

from LightGlue.lightglue import LightGlue, SuperPoint, SIFT
from LightGlue.lightglue.utils import load_image, rbd
from LightGlue.lightglue import viz2d

from SuperGluePretrainedNetwork import match_pairs as SuperGlueMatching

from omniglue import omniglue_extract as omniglue

import cv2


def get_dataset(dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    Load dataset and split into training, validation, and test sets.

    Args:
        dataset_name (str): Name of the dataset.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]: DataFrames for training, validation, and test sets, and the dataset root directory.
    """
    if dataset_name == "LeopardID2022":
        d = datasets.LeopardID2022("datasets/LeopardID2022")
    elif dataset_name == "HyenaID2022":
        d = datasets.HyenaID2022("datasets/HyenaID2022")
    elif dataset_name == "ATRW":
        d = datasets.ATRW("datasets/ATRW")
    else:
        raise ValueError("Invalid dataset name")
    
    # remove those samples where an identity has fewer than 5 samples
    d.df = d.df.groupby("identity").filter(lambda x: len(x) >= 5)

    if dataset_name == "ATRW":
        df_test = d.df[d.df["original_split"] == "test"]
        df_train = d.df[d.df["original_split"] == "train"]
        df_val = df_test
    else:
        n_identites = len(d.df['identity'].unique())

        n_test_ids = int(np.ceil(0.33 * n_identites))
        n_val_ids = int(np.ceil(0.2 * (n_identites - n_test_ids)))

        splitter = splits.DisjointSetSplit(n_class_test=n_test_ids) # 64 test IDs = ceil(0.33 * 193)
        # splitter = splits.ClosedSetSplit(0.67)
        for idx_train, idx_test in splitter.split(d.df):
            _df_train, df_test = d.df.loc[idx_train], d.df.loc[idx_test]

        splitter_2 = splits.DisjointSetSplit(n_class_test=n_val_ids) # 26 val IDs = ceil(0.2 * 129)
        for idx_train, idx_val in splitter_2.split(_df_train):
            df_train, df_val = _df_train.loc[idx_train], _df_train.loc[idx_val]


    df_train = df_train.groupby("identity").filter(lambda x: len(x) >= 5)
    df_test = df_test.groupby("identity").filter(lambda x: len(x) >= 5)
    df_val = df_val.groupby("identity").filter(lambda x: len(x) >= 5)

    return df_train, df_val, df_test, d.root


def get_query_gallery_split(df: pd.DataFrame, root: str, transform: T.Compose, dataset_name: str) -> Tuple[WildlifeDataset, WildlifeDataset]:
    """
    Split dataset into query and gallery sets. For each identity, the first sample is used as the gallery image and the rest are used as query images.

    Args:
        df (pd.DataFrame): DataFrame containing dataset information.
        root (str): Root directory of the dataset.
        transform (T.Compose): Transformations to apply to the images.
        dataset_name (str): Name of the dataset.

    Returns:
        Tuple[WildlifeDataset, WildlifeDataset]: Query and gallery datasets.
    """
    df_query = df.groupby("identity").apply(lambda x: x.iloc[1:])
    df_gallery = df.groupby("identity").apply(lambda x: x.iloc[:1])
    if dataset_name == "ATRW":
        dataset_query = WildlifeDataset(df_query, root, transform=transform)
        dataset_gallery = WildlifeDataset(df_gallery, root, transform=transform)
    else:    
        dataset_query = WildlifeDataset(df_query, root, transform=transform, img_load="bbox")
        dataset_gallery = WildlifeDataset(df_gallery, root, transform=transform, img_load="bbox")
    return dataset_query, dataset_gallery


def get_matches_megadesc(dataset_query: WildlifeDataset, dataset_gallery: WildlifeDataset, device: torch.device) -> np.ndarray:
    """
    Get top 5 matches using MegaDescriptor.

    Args:
        dataset_query (WildlifeDataset): Query image dataset.
        dataset_gallery (WildlifeDataset): Gallery image dataset.
        device (torch.device): Device to run the model on.

    Returns:
        np.ndarray (n_query, 5): Indices of the top 5 matches for each query image.
    """
    # Load the MegaDepth model
    model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)
    model = model.eval().to(device)

    # Load the MegaDepth descriptors
    features = DeepFeatures(model, device=device)

    # Compute the MegaDepth descriptors for the query and gallery images
    query_desc = features(dataset_query)
    gallery_desc = features(dataset_gallery)

    # Compute the cosine similarity between the query and gallery descriptors
    similarity = CosineSimilarity()
    similarity_matrix = similarity(query_desc, gallery_desc)['cosine']

    # Find the top 5 matches
    # knn = KnnClassifier(k=1)
    # matches = knn(similarity_matrix)
    top_k = 5
    matches = np.argsort(similarity_matrix, axis=1)[:, -top_k:][:, ::-1]  # Sort and take top 5, then reverse order

    return matches


def get_matches_lightglue(image0: torch.Tensor, image1: torch.Tensor, device: torch.device, detector: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Get matches using LightGlue.

    Args:
        image0 (torch.Tensor): First image tensor.
        image1 (torch.Tensor): Second image tensor.
        device (torch.device): Device to run the model on.
        detector (str): Keypoint detector to use.

    Returns:
        Tuple[np.ndarray, np.ndarray, dict]: Points from the first and second images and matches.
    """
    if detector == "superpoint":
        extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
    elif detector == "sift":
        extractor = SIFT(max_num_keypoints=2048).eval().cuda()
    else:
        raise ValueError(f"Invalid keypoint detector: {detector}")
    
    matcher = LightGlue(features=detector).eval().cuda()  # load the matcher

    image0 = image0.to(device)
    image1 = image1.to(device)

    # import pdb; pdb.set_trace()
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)

    # match the features
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
    matches = matches01['matches']  # indices with shape (K,2)
    points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
    points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

    indices = matches01['scores'] > 0.05
    points0, points1 = points0[indices], points1[indices]
    matches01['matches'] = matches01['matches'][indices]
    matches01['scores'] = matches01['scores'][indices]

    # Visualization
    image0 = image0.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image1 = image1.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # axes = viz2d.plot_images([image0, image1])
    # viz2d.plot_matches(points0, points1, color="lime", lw=0.2)
    # viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)

    # kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
    # viz2d.plot_images([image0, image1])
    # viz2d.plot_keypoints([points0, points1], colors=[kpc0, kpc1], ps=10)

    return points0, points1, matches01


def get_matches_superglue(image0: torch.Tensor, image1: torch.Tensor, device: torch.device, detector: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get matches using SuperGlue.

    Args:
        image0 (torch.Tensor): First image tensor.
        image1 (torch.Tensor): Second image tensor.
        device (torch.device): Device to run the model on.
        detector (str): Keypoint detector to use.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Points from the first and second images and match scores.
    """
    if detector == "superpoint":
        extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
    # elif detector == "sift":
    #     extractor = SIFT(max_num_keypoints=2048).eval().cuda()
    else:
        raise ValueError(f"Invalid keypoint detector: {detector}")


    image0 = image0.to(device)
    image1 = image1.to(device)

    # import pdb; pdb.set_trace()
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)

    # Remove the batch dimension and convert to NumPy array
    image0 = image0.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image0 = (image0 * 255).astype(np.uint8)
    image0 = cv2.cvtColor(image0, cv2.COLOR_RGB2GRAY)

    image1 = image1.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image1 = (image1 * 255).astype(np.uint8)
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

    sys.stdout = open(os.devnull, "w")
    points0, points1, match_scores = SuperGlueMatching.run(img0=image0, kpts0=feats0, img1=image1, kpts1=feats1, match_thresold=0.2)
    sys.stdout = sys.__stdout__
    return points0, points1, match_scores
    

def get_matches_omniglue(image0: torch.Tensor, image1: torch.Tensor, device: torch.device, detector: str, omniglue_matcher: omniglue.OmniGlue) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get matches using OmniGlue.

    Args:
        image0 (torch.Tensor): First image tensor.
        image1 (torch.Tensor): Second image tensor.
        device (torch.device): Device to run the model on.
        detector (str): Keypoint detector to use.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Points from the first and second images and match scores.
    """
    if detector == "superpoint":
        extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
    # elif detector == "sift":
    #     extractor = SIFT(max_num_keypoints=2048).eval().cuda()
    else:
        raise ValueError(f"Invalid keypoint detector: {detector}")

    image0 = image0.to(device)
    image1 = image1.to(device)

    # import pdb; pdb.set_trace()
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)

    feats0['keypoints'] = feats0['keypoints'].cpu().numpy()
    feats0['keypoint_scores'] = feats0['keypoint_scores'].cpu().numpy()
    feats0['descriptors'] = feats0['descriptors'].cpu().numpy()
    feats1['keypoints'] = feats1['keypoints'].cpu().numpy()
    feats1['keypoint_scores'] = feats1['keypoint_scores'].cpu().numpy()
    feats1['descriptors'] = feats1['descriptors'].cpu().numpy()

    # Remove the batch dimension and convert to NumPy array
    image0 = image0.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image0 = (image0 * 255).astype(np.uint8)

    image1 = image1.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image1 = (image1 * 255).astype(np.uint8)

    match_kp0s, match_kp1s, match_confidences = omniglue_matcher.FindMatches(image0, image1, kpts0=feats0, kpts1=feats1, match_threshold=0.05)
#    import pdb; pdb.set_trace()
    return match_kp0s, match_kp1s, match_confidences


def run_inference(dataset_query: WildlifeDataset, dataset_gallery: WildlifeDataset, device: torch.device, detector: str, matcher: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inference pipeline for feature matching.
    Computes the top 5 matches for each query image and computes the top-1, top-3, and top-5 accuracy.

    Args:
        dataset_query (WildlifeDataset): Query image dataset.
        dataset_gallery (WildlifeDataset): Gallery image dataset.
        device (torch.device): Device to run the model on.
        detector (str): Keypoint detector to use.
        matcher (str): Feature matcher to use.

    Returns:
        Tuple[np.ndarray, np.ndarray]
            - np.ndarray: Predictions (N, 5)
            - np.ndarray: Ground truth (N,)
    """

    if matcher == "megadescriptor":
        top_5_matches = get_matches_megadesc(dataset_query, dataset_gallery, device)
        predictions = np.array(top_5_matches)
        gt = np.array([dataset_query[i][1] for i in range(len(dataset_query))])
        return predictions, gt

    if matcher == "omniglue":
            og = omniglue.OmniGlue(
            og_export='omniglue/models/og_export',
            sp_export='omniglue/models/sp_v6',
            dino_export='omniglue/models/dinov2_vitb14_pretrain.pth',
        )

    predictions = np.zeros((len(dataset_query), 5)) # store the indices of the top 5 IDs (decreasing order of matches)
    gt = np.zeros(len(dataset_query)) # store the ground truth IDs

    for i in tqdm(range(len(dataset_query)), desc="Querying"):
        image_query = dataset_query[i][0]
        image_query = image_query.unsqueeze(0).to(device)

        # for each image in dataset_gallery, store the number of matches
        n_matches_list = np.zeros(len(dataset_gallery))

        for j in range(len(dataset_gallery)):
            image_gallery = dataset_gallery[j][0]
            image_gallery = image_gallery.unsqueeze(0).to(device)

            if matcher == "lightglue":
                pts_query, pts_gallery, matches = get_matches_lightglue(image_query, image_gallery, device, detector=detector)
                n_matches = len(matches["matches"])
            elif matcher == "superglue":
                pts_query, pts_gallery, matches = get_matches_superglue(image_query, image_gallery, device, detector=detector)
                n_matches = len(matches)
            elif matcher == "omniglue":
                pts_query, pts_gallery, matches = get_matches_omniglue(image_query, image_gallery, device, detector=detector, omniglue_matcher=og)
                n_matches = len(matches)
            elif matcher == "megadescriptor":
                n_matches = get_matches_megadesc(image_query, image_gallery, device)
            else:
                raise ValueError(f"Invalid feature matcher {matcher}")
            n_matches_list[j] = n_matches
       
        # get the top 5 matches
        top_5_matches = np.argsort(n_matches_list)[-5:][::-1]
        top_5_matches = [dataset_gallery[i][1] for i in top_5_matches]

        predictions[i] = np.array(top_5_matches)
        gt[i] = dataset_query[i][1]

    return predictions, gt


def compute_metrics(preds: np.ndarray, gt: np.ndarray) -> None:
    """
    Compute top-1, top-3, and top-5 accuracy, precision, and recall.
    
    Args:
        preds (np.ndarray): (N, 5) Predictions.
        gt (np.ndarray): (N,) Ground truth.
    """

    # compute top-k accuracy
    top_1 = np.sum(preds[:, 0] == gt) / len(gt)
    top_3 = np.sum([gt[i] in preds[i, :3] for i in range(len(gt))]) / len(gt)
    top_5 = np.sum([gt[i] in preds[i, :] for i in range(len(gt))]) / len(gt)

    # compute precision and recall for k=1, 3, 5
    TP = np.sum( preds[:, 0] == gt )
    FP = np.sum( preds[:, 0] != gt )
    FN = len(gt) - TP
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    print("------k=1------")
    print(f"Top-1 Accuracy: {100*top_1:.2f}%")
    print(f"Precision: {100*precision:.2f}%")
    print(f"Recall: {100*recall:.2f}%")
    print(F"TP: {TP}, FP: {FP}, FN: {FN}")

    # k = 3
    TP = np.sum([gt[i] in preds[i, :3] for i in range(len(gt))])
    FP = np.sum([gt[i] not in preds[i, :3] for i in range(len(gt))])
    FN = len(gt) - TP
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    print("------k=3------")
    print(f"Top-3 Accuracy: {100*top_3:.2f}%")
    print(f"Precision: {100*precision:.2f}%")
    print(f"Recall: {100*recall:.2f}%")
    print(F"TP: {TP}, FP: {FP}, FN: {FN}")

    # k = 5
    TP = np.sum([gt[i] in preds[i, :] for i in range(len(gt))])
    FP = np.sum([gt[i] not in preds[i, :] for i in range(len(gt))])
    FN = len(gt) - TP
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    print("------k=5------")
    print(f"Top-5 Accuracy: {100*top_5:.2f}%")
    print(f"Precision: {100*precision:.2f}%")
    print(f"Recall: {100*recall:.2f}%")
    print(F"TP: {TP}, FP: {FP}, FN: {FN}")


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    # specify detector either superpoint or sift. Default to superpoint. Only allow these two options.
    parser.add_argument("--detector", type=str, default="superpoint", help="Detector to use for feature extraction", choices=["superpoint", "sift", "megadescriptor"])
    parser.add_argument("--matcher", type=str, help="Matcher to use for feature matching", choices=["megadescriptor", "lightglue", "superglue", "omniglue"])
    parser.add_argument("--dataset", type=str, help="Dataset to use for inference", choices=["LeopardID2022", "HyenaID2022", "ATRW"])
    return parser.parse_args()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df_train, df_val, df_test, root = get_dataset(args.dataset)
    print(f"Number of training samples: {len(df_train)}")
    print(f"Number of validation samples: {len(df_val)}")
    print(f"Number of test samples: {len(df_test)}")

    transforms = T.Compose([
        T.Resize([384, 384]),
        T.ToTensor(),
        # T.Normalize(
        #     [0.5, 0.5, 0.5],
        #     [0.5, 0.5, 0.5]
        # )
    ])
    print("Creating query and gallery datasets...")
    dataset_query, dataset_gallery = get_query_gallery_split(df_val, root, transforms, dataset_name=args.dataset)
    print(f"Number of query samples: {len(dataset_query)}")
    print(f"Number of gallery samples: {len(dataset_gallery)}")

    preds, gt = run_inference(dataset_query, dataset_gallery, device, detector=args.detector, matcher=args.matcher)

    save_dir = f"/home/atharv21027/reid-graphs/outputs/{args.matcher}/{args.dataset}/"
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"{args.detector}_preds.npy"), preds)
    np.save(os.path.join(save_dir, f"{args.detector}_gt.npy"), gt)
    print(f"Predictions saved at {save_dir}")

    compute_metrics(preds, gt)


if __name__ == "__main__":
    args = parse_args()
    main(args)
