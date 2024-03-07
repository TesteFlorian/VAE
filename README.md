# VAE for Dimension Reduction in Satellite Time Series

This repository contains an implementation of a Variational Autoencoder (VAE) neural network designed for dimension reduction in satellite time series data. 

## Introduction

Variational Autoencoders are powerful generative models that learn to represent complex high-dimensional data in a lower-dimensional latent space. They are particularly useful for tasks such as dimensionality reduction, data compression, and feature extraction. In the context of satellite time series data, VAEs can be employed to capture meaningful patterns and extract useful characteristics from the data.

## Usage

To use the VAE implemented in this repository, follow these steps:

1. Install the necessary dependencies by running:
    ```
    pip install -r requirements.txt
    ```

2. Execute the Python script `vae_train.py` with the desired parameters. For example:
    ```
    python vae_train.py --exp vae --dataset GLASS --lr 1e-4 --num_epochs 250 --batch_size 32 --batch_size_test 8 --z_dim 10 --beta 0.1 --nb_channels 1 --model vae --force_train --month dGPP_May
    ```

3. The script will train the VAE model on the specified dataset and parameters. Once trained, the model can be used for dimensionality reduction and extracting useful characteristics from satellite time series data.

## References

If you use this code in your research or project, please consider citing the following paper:
[Insert paper citation here]

## License

This project is licensed under the [MIT License](LICENSE).

