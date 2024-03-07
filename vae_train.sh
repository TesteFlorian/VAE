#!/bin/bash
set -euxo pipefail
# Define list of beta values to try
betas=(0.001 0.01 0.1 1 2 5 10)

# Loop over values of z_dim
for month in "dGPP_May" "dGPP_June" "dGPP_July" "dGPP_August" "dGPP_September"; do
    for ((z_dim = 1; z_dim <= 36; z_dim++)); do
        # Loop over beta values
        for beta in "${betas[@]}"; do
            # Execute Python script with specified parameters
            python vae_train.py \
                --exp vae \
                --dataset GLASS \
                --lr 1e-4 \
                --num_epochs 250 \
                --batch_size 32 \
                --batch_size_test 8 \
                --z_dim "$z_dim" \
                --beta "$beta" \
                --nb_channels 1 \
                --model vae \
                --force_train \
                --month "$month"
        done
    done
done