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

2. For Windows Users:
    - Use the provided PowerShell script (`vae_train.ps1`) to define the desired parameters for running the script. Open a PowerShell terminal and navigate to the repository directory. Then, execute the following command to run the script:
        ```powershell
        .\vae_train.ps1
        ```

3. For Linux Users:
    - Use the provided Bash script (`vae_train.sh`) to define the desired parameters for running the script. Open a terminal and navigate to the repository directory. Then, execute the following command to run the script:
        ```bash
        bash vae_train.sh
        ```
'. The script will train the VAE model on the specified dataset and parameters. The model can be used for dimensionality reduction and extracting useful characteristics from satellite time series data. The model learns to compress the input data into a lower-dimensional latent space while preserving the essential features. These extracted features can then be used for downstream tasks such as clustering and classification.

## References

If you use this code in your research or project, please consider citing the following paper:
[Insert paper citation here]

## License

This project is licensed under the [MIT License](LICENSE).

