# ReID-with-graphs

Code for running and evaluating baseline experiments for Wildlife ReID using graph neural network (GNN)-based methods. The code uses the SuperGlue, LightGlue, and OmniGlue models for feature matching and the SIFT and SuperPoint for feature extraction. We use MegaDetector model for object detection. 

We also use MegaDescriptor and ArcFace from the InsightFace repository to train an end-to-end model for feature extraction and matching.

The code is used to run experiments on the ATRW, LeopardID2022, HyenaID2022, and WII datasets.

# Directory Structure
```
.
├── compute_metrics.py
├── datasets/
│   ├── ATRW/
│   ├── LeopardID2022/
│   └── HyenaID2022/
├── flank-classifier/
├── insightface/
├── LightGlue/
├── MegaDetector/
├── notebooks/
├── omniglue/
├── replacements/
├── run.py
├── SuperGluePretrainedNetwork/
├── wii_dataset.py
├── wildlife-datasets/
└── wildlife-tools/
```

# Setting up the environment

1. Set up the environment:
```bash
conda env create --file conda-env.yml
conda activate wildlife10
```

2. Clone the relevant repositories:
```bash
git clone https://github.com/google-research/omniglue.git
git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git
git clone https://github.com/cvg/LightGlue.git
```

3. Initialize the repositories and our modifications to them:
```bash
pip install -e omniglue/
pip install -e LightGlue/
pip install insightface
cp replacements/sg_utils.py SuperGluePretrainedNetwork/models/utils.py
cp replacements/sg_match_pairs.py SuperGluePretrainedNetwork/match_pairs.py
cp replacements/og_extract.py omniglue/src/omniglue/omniglue_extract.py
```

# Running the experiments

Run the experiments by running the following command:
```bash
python3 run.py --detector [superpoint, sift, megadescriptor, arcface] --matcher [superglue, lightglue, omniglue, megadescriptor, arcface] --dataset [HyenaID2022, LeopardID2022, ATRW, WII]
```

Predictions will be saved in the `outputs/` directory.

# Computing metrics
To compute and display evaluation metrics from the saved prediction files, run the same command as above but invoke `compute_metrics.py` instead of `run.py`.

# Disclaimer

`wildlife-tools` is our own fork of the [WildlifeDatasets repository](https://github.com/WildlifeDatasets/wildlife-tools) with some minor modifications. We have included it in this repository for convenience.
