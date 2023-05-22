#!/bin/bash

python3 vae_train.py\
    --exp=test_vae_grf\
    --dataset=GLASS\
    --lr=1e-3\
    --num_epochs=1\
    --batch_size=16\
    --batch_size_test=16\
    --z_dim=37\
    --beta=1\
    --nb_channels=1\
    --model=vae\
    --force_train\

