#!/bin/bash

python3 vae_train.py\
    --exp=test_vae_grf\
    --dataset=livestock\
    --category=wood\
    --lr=1e-2\
    --num_epochs=200\
    --img_size=256\
    --batch_size=16\
    --batch_size_test=8\
    --z_dim=38\
    --beta=1\
    --nb_channels=3\
    --model=vae\
    --force_train\

