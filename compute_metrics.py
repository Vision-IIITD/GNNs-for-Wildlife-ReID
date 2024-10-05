from wildlife_datasets import datasets, splits
from wildlife_tools.data import WildlifeDataset
# from wildlife_tools.features import SIFTFeatures

import os
import torchvision.transforms as T
import numpy as np
from typing import Tuple, List
import pandas as pd
from argparse import ArgumentParser


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
    elif dataset_name == "WII":
        import wii_dataset
        d = wii_dataset.WII("datasets/WII")
    else:
        raise ValueError("Invalid dataset name")
    
    # remove those samples where an identity has fewer than 5 samples
    d.df = d.df.groupby("identity").filter(lambda x: len(x) >= 5)

    if dataset_name == "ATRW":
        df_train = d.df[d.df["original_split"] == "train"]
        df_test = d.df[d.df["original_split"] == "test"]
        df_val = df_test

    elif dataset_name == "WII":
        df_train = d.df[d.df["split"] == "train"]
        df_test = d.df[d.df["split"] == "test"]

        # remove those entries that dont have a flank column
        df_test = df_test.dropna(subset=["flank"])
        df_test = df_test.dropna(subset=["gallery_split"])

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

    if dataset_name == "WII":
        df_left = df[df["flank"] == "left"]
        df_right = df[df["flank"] == "right"]

        df_left_query = df_left[df_left["gallery_split"] == "query"]
        df_left_gallery = df_left[df_left["gallery_split"] == "gallery"]

        df_right_query = df_right[df_right["gallery_split"] == "query"]
        df_right_gallery = df_right[df_right["gallery_split"] == "gallery"]

        dataset_left_query = WildlifeDataset(df_left_query, root, transform=transform, img_load="bbox")
        dataset_left_gallery = WildlifeDataset(df_left_gallery, root, transform=transform, img_load="bbox")

        dataset_right_query = WildlifeDataset(df_right_query, root, transform=transform, img_load="bbox")
        dataset_right_gallery = WildlifeDataset(df_right_gallery, root, transform=transform, img_load="bbox")

        return dataset_left_query, dataset_left_gallery, dataset_right_query, dataset_right_gallery
    else:
        df_query = df.groupby("identity").apply(lambda x: x.iloc[1:])
        df_gallery = df.groupby("identity").apply(lambda x: x.iloc[:1])

    if dataset_name == "ATRW":
        dataset_query = WildlifeDataset(df_query, root, transform=transform)
        dataset_gallery = WildlifeDataset(df_gallery, root, transform=transform)
    else:    
        dataset_query = WildlifeDataset(df_query, root, transform=transform, img_load="bbox")
        dataset_gallery = WildlifeDataset(df_gallery, root, transform=transform, img_load="bbox")
    return dataset_query, dataset_gallery


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    # specify detector either superpoint or sift. Default to superpoint. Only allow these two options.
    parser.add_argument("--detector", type=str, default="superpoint", help="Detector to use for feature extraction", choices=["superpoint", "sift", "megadescriptor", "arcface"])
    parser.add_argument("--matcher", type=str, help="Matcher to use for feature matching", choices=["megadescriptor", "lightglue", "superglue", "omniglue", "arcface"])
    parser.add_argument("--dataset", type=str, help="Dataset to use for inference", choices=["LeopardID2022", "HyenaID2022", "ATRW", "WII"])
    return parser.parse_args()


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
    # print(f"Precision: {100*precision:.2f}%")
    # print(f"Recall: {100*recall:.2f}%")
    # print(F"TP: {TP}, FP: {FP}, FN: {FN}")
    print(f"Correct Matches: {TP}, Incorrect Matches: {FP}")

    # k = 3
    TP = np.sum([gt[i] in preds[i, :3] for i in range(len(gt))])
    FP = np.sum([gt[i] not in preds[i, :3] for i in range(len(gt))])
    FN = len(gt) - TP
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    print("------k=3------")
    print(f"Top-3 Accuracy: {100*top_3:.2f}%")
    # print(f"Precision: {100*precision:.2f}%")
    # print(f"Recall: {100*recall:.2f}%")
    # print(F"TP: {TP}, FP: {FP}, FN: {FN}")
    print(f"Correct Matches: {TP}, Incorrect Matches: {FP}")

    # k = 5
    TP = np.sum([gt[i] in preds[i, :] for i in range(len(gt))])
    FP = np.sum([gt[i] not in preds[i, :] for i in range(len(gt))])
    FN = len(gt) - TP
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    print("------k=5------")
    print(f"Top-5 Accuracy: {100*top_5:.2f}%")
    # print(f"Precision: {100*precision:.2f}%")
    # print(f"Recall: {100*recall:.2f}%")
    # print(f"TP: {TP}, FP: {FP}, FN: {FN}")
    print(f"Correct Matches: {TP}, Incorrect Matches: {FP}")


def main(args):    
    # df_train, df_val, df_test, root = get_dataset(args.dataset)

    # transforms = T.Compose([
    #     T.Resize([384, 384]),
    #     T.ToTensor(),
    #     # T.Normalize(
    #     #     [0.5, 0.5, 0.5],
    #     #     [0.5, 0.5, 0.5]
    #     # )
    # ])
    # dataset_query, dataset_gallery = get_query_gallery_split(df_val, root, transforms)
    # print(f"Number of query samples: {len(dataset_query)}")
    # print(f"Number of gallery samples: {len(dataset_gallery)}")

    save_dir = f"/home/atharv21027/reid-graphs/outputs/{args.matcher}/{args.dataset}/"
    save_dir = f"./"

    path_preds = os.path.join(save_dir, f"preds_left.npy")
    path_gt = os.path.join(save_dir, f"gt_left.npy")
    path_preds_r = os.path.join(save_dir, f"preds_right.npy")
    path_gt_r = os.path.join(save_dir, f"gt_right.npy")
    if not os.path.exists(path_preds) or not os.path.exists(path_gt):
        raise FileNotFoundError("Predictions and/or ground truth not found. Please run inference first.")
    preds_l = np.load(path_preds)
    gt_l = np.load(path_gt)
    preds_r = np.load(path_preds_r)
    gt_r = np.load(path_gt_r)

    preds = np.concatenate([preds_l, preds_r])
    gt = np.concatenate([gt_l, gt_r])

    compute_metrics(preds, gt)


if __name__ == "__main__":
    args = parse_args()
    main(args)
