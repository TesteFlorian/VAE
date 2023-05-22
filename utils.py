import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision
import torch
from torch.utils.data import DataLoader
from datasets import *
from vae import VAE
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_id", default=100)
    parser.add_argument("--img_size", default=270, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--batch_size_test", default=16, type=int)
    parser.add_argument("--num_epochs", default=2000, type=int)
    parser.add_argument("--latent_img_size", default=32, type=int)
    parser.add_argument("--z_dim", default=38, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--beta", default=1.0, type=float)
    parser.add_argument("--exp", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--dataset", default="GLASS")
    parser.add_argument("--category", default=None)
    parser.add_argument("--defect", default=None)
    parser.add_argument(
        "--defect_list",
        type=lambda s: [item for item in s.split(',')]
    )
    parser.add_argument("--rec_loss", default="xent")
    parser.add_argument("--nb_channels", default=3, type=int)
    parser.add_argument("--model", default="vae_grf")
    parser.add_argument("--corr_type", default="corr_exp")
    parser.add_argument("--force_train", dest='force_train', action='store_true')
    parser.set_defaults(force_train=False)
    parser.add_argument("--force_cpu", dest='force_cpu', action='store_true')
    parser.set_defaults(force_train=False)

    return parser.parse_args()

def load_vqvae(args):
    if args.model == "vae":
        model = VAE(
                z_dim=args.z_dim,
                img_size=args.img_size,
                nb_channels=args.nb_channels,
                beta=args.beta,
            )

    return model

def load_model_parameters(model, file_name, dir1, dir2, device):
    print(f"Trying to load: {file_name}")
    try:
        state_dict = torch.load(
            os.path.join(dir1, file_name),
            map_location=device
        )
    except FileNotFoundError:
        state_dict = torch.load(
            os.path.join(dir2, file_name),
            map_location=device
        )
    model.load_state_dict(state_dict, strict=False)
    print(f"{file_name} loaded !")

    return model

def get_train_dataloader(args):
    if args.dataset == "GLASS":
        train_dataset = GLASS_TrainDataset(
            args.img_size
        )
    else:
        raise RuntimeError("No / Wrong dataset provided")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
        shuffle=False if args.dataset == "ssl_vqvae" else True)

    return train_dataloader, train_dataset

def get_test_dataloader(args,):
    if args.dataset == "GLASS":
        test_dataset = GLASS_TestDataset(
            args.img_size
        )
    else:
        raise RuntimeError("No / Wrong dataset provided")

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_test,
        )

    return test_dataloader, test_dataset

def tensor_img_to_01(t, share_B=False):
    ''' t is a BxCxHxW tensor, put its values in [0, 1] for each batch element
    if share_B is False otherwise normalization include all batch elements
    '''
    t = torch.nan_to_num(t)
    if share_B:
        t = ((t - torch.amin(t, dim=(0, 1, 2, 3), keepdim=True)) /
            (torch.amax(t, dim=(0, 1, 2, 3), keepdim=True) - torch.amin(t,
            dim=(0, 1, 2,3),
            keepdim=True)))
    if not share_B:
        t = ((t - torch.amin(t, dim=(1, 2, 3), keepdim=True)) /
            (torch.amax(t, dim=(1, 2, 3), keepdim=True) - torch.amin(t, dim=(1, 2,3),
            keepdim=True)))
    return t

def update_loss_dict(ld_old, ld_new):
    for k, v in ld_new.items():
        if k in ld_old:
            ld_old[k] += v
        else:
            ld_old[k] = v
    return ld_old

def print_loss_logs(f_name, out_dir, loss_dict, epoch, exp_name):
    if epoch == 0:
        with open(f_name, "w") as f:
            print("epoch,", end="", file=f)
            for k, v in loss_dict.items():
                print(f"{k},", end="", file=f)
            print("\n", end="", file=f)
    # then, at every epoch
    with open(f_name, "a") as f:
        print(f"{epoch + 1},", end="", file=f)
        for k, v in loss_dict.items():
            print(f"{v},", end="", file=f)
        print("\n", end="", file=f)
    if (epoch + 1) % 50 == 0 or epoch in [4, 9, 24]:
        # with this delimiter one spare column will be detected
        arr = np.genfromtxt(f_name, names=True, delimiter=",")
        fig, axis = plt.subplots(1)
        for i, col in enumerate(arr.dtype.names[1:-1]):
            axis.plot(arr[arr.dtype.names[0]], arr[col], label=col)
        axis.legend()
        fig.savefig(os.path.join(out_dir,
            f"{exp_name}_loss_{epoch + 1}.tif"))
        plt.close(fig) 
