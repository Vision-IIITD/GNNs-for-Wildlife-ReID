# ReID-with-graphs

Code for running and evaluating baseline experiments.

# Directory Structure
```
.
├── compute_metrics.py
├── datasets/
│   ├── ATRW/
│   ├── LeopardID2022/
│   └── HyenaID2022/
├── LightGlue/
├── omniglue/
├── replacements/
└── run.py
├── SuperGluePretrainedNetwork/
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
pip install -e SuperGluePretrainedNetwork/
pip install -e LightGlue/
mv replacements/sg_utils.py SuperGluePretrainedNetwork/models/utils.py
mv replacements/sg_match_pairs.py SuperGluePretrainedNetwork/match_pairs.py
mv replacements/og_extract.py omniglue/src/omniglue/omniglue_extract.py
```

# Running the experiments

Run the experiments by running the following command:
```bash
python3 run.py --detector [superpoint, sift, megadescriptor] --matcher [superglue, lightglue, omniglue, megadescriptor] --dataset [HyenaID2022, LeopardID2022, ATRW]
```

Predictions will be saved in the `outputs/` directory.

# Computing metrics
To compute and display evaluation metrics from the saved prediction files, run the same command as above but invoke `compute_metrics.py` instead of `run.py`.