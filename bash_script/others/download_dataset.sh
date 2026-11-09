#!/bin/bash
# Original robomimic low-dim datasets, for the replay path (README option 2).
# The preprocessed demos are also available directly; see download_data.sh.
mkdir -p data && cd data
wget https://diffusion-policy.cs.columbia.edu/data/training/robomimic_lowdim.zip
unzip robomimic_lowdim.zip && rm -f robomimic_lowdim.zip
cd ..
