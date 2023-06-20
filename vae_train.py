import os
import argparse
import numpy as np
import time
import pandas as pd

from torchvision import transforms, utils
import torch
from torch import nn

torch.backends.cudnn.benchmark = True

from utils import (get_train_dataloader,
                   get_test_dataloader,
                   load_model_parameters,
                   load_vqvae,
                   update_loss_dict,
                   print_loss_logs,
                   parse_args
                   )

import matplotlib.pyplot as plt

import sys

def train(model, train_loader, device, optimizer, epoch):

    model.train()
    train_loss = 0
    loss_dict = {}

    for batch_idx, (input_mb) in enumerate(train_loader):
        print(batch_idx + 1, end=", ", flush=True)
        input_mb = input_mb.to(device) 
        # lbl = lbl.to(device) 
        optimizer.zero_grad() # otherwise grads accumulate in backward

        loss, recon_mb, loss_dict_new = model.step(
            input_mb
        )

        (-loss).backward()
        train_loss += loss.item()
        loss_dict = update_loss_dict(loss_dict, loss_dict_new)
        optimizer.step()
    
    nb_mb_it = (len(train_loader.dataset) // input_mb.shape[0])
    train_loss /= nb_mb_it
    loss_dict = {k:v / nb_mb_it for k, v in loss_dict.items()}
    return train_loss, input_mb, recon_mb, loss_dict


def eval(model, test_loader, device):
    model.eval()
    input_mb = next(iter(test_loader))
    # gt_mb = gt_mb.to(device)
    input_mb = input_mb.to(device)
    recon_mb, opt_out = model(input_mb)
    recon_mb = model.mean_from_lambda(recon_mb)
    return input_mb, recon_mb, opt_out


def extract_mu_values(model, train_dataset_tensor, device_extraction="cpu", epoch=""):
    model.eval()
    model.to(device_extraction)
    #for batch_data in data_loader:
    #    
    #    batch_data = batch_data.to(device_extraction)  
    #    # output = model.encoder(batch_data)
    #    mu_values, logvar = model.encoder(batch_data)
    print("train_dataset[:] shape", train_dataset_tensor.shape)
    mu_values, _ = model.encoder(train_dataset_tensor)    
    print("mu_values shape", mu_values.shape)

    # mu_values = torch.cat(mu_values, dim=0)
    # Convert the batch to a NumPy array
    mu_values_np = mu_values.detach().cpu().numpy()
    mu_values_np = np.mean(mu_values_np,axis=(1))
    print(f"The mu shape is {mu_values_np.shape}")
    # Export to csv
    directory = './torch_results'
    os.makedirs(directory, exist_ok=True)

    file_path = os.path.join(directory, f'mu_values_{epoch}.csv')

    np.savetxt(file_path, mu_values_np, delimiter=',')
    print(f"Mu values saved to: {file_path}")

    return mu_values, mu_values_np 
   


def main(args):
    if torch.cuda.is_available() and not args.force_cpu:
        device = torch.device("cuda")
        print("Number of CUDA devices:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            print(f"Device {i}: {device_name}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available or --force-cpu is set.")

    print("PyTorch device:", device)

    
    seed = 11
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    model = load_vqvae(args)
    model.to(device)

    train_dataloader, train_dataset = get_train_dataloader(args)
    test_dataloader, test_dataset = get_test_dataloader(args)
    train_dataset_tensor = []
    for i in range(len(train_dataset)):
        train_dataset_tensor.append(torch.from_numpy(train_dataset[i]))
    train_dataset_tensor = torch.stack(train_dataset_tensor,axis = 0)
    nb_channels = args.nb_channels

    img_size = args.img_size
    batch_size = args.batch_size
    batch_size_test = args.batch_size_test

    print("Nb channels", nb_channels, "img_size", img_size, 
        "mini batch size", batch_size)


    out_dir = './torch_logs'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    checkpoints_dir ="./torch_checkpoints"
    if not os.path.isdir(checkpoints_dir):
        os.mkdir(checkpoints_dir)
    checkpoints_saved_dir ="./torch_checkpoints_saved"
    res_dir = './torch_results'
    if not os.path.isdir(res_dir):
        os.mkdir(res_dir)
    data_dir = './torch_datasets'
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)


    try:
        if args.force_train:
            raise FileNotFoundError
        file_name = f"{args.exp}_{args.params_id}.pth"
        model = load_model_parameters(model, file_name, checkpoints_dir,
            checkpoints_saved_dir, device)
    except FileNotFoundError:
        print("Starting training")
        #print([p for p in model.parameters()])
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr
        )
        for epoch in range(args.num_epochs):
            print("Epoch", epoch + 1)
            loss, input_mb, recon_mb, loss_dict,  = train(
                model=model,
                train_loader=train_dataloader,
                device=device,
                optimizer=optimizer,
                epoch=epoch)
            print('epoch [{}/{}], train loss: {:.4f}'.format(
                epoch + 1, args.num_epochs, loss))

            # print loss logs
            f_name = os.path.join(out_dir, f"{args.exp}_loss_values.txt")
            print_loss_logs(f_name, out_dir, loss_dict, epoch, args.exp)

            #print(input_mb[0])
            #print(recon_mb[0])
            #plt.imshow(np.moveaxis(recon_mb[0].detach().cpu().numpy(), 0, 2))
            #plt.show()
                    
            # save model parameters
            if (epoch + 1) % 100 == 0 or epoch in [0, 4, 9, 24]:
                # to resume a training optimizer state dict and epoch
                # should also be saved
                torch.save(model.state_dict(), os.path.join(
                    checkpoints_dir, f"{args.exp}_{epoch + 1}.pth"
                    )
                )
           
             

            # print some reconstrutions
            if (epoch + 1) % 50 == 0 or epoch in [0, 4, 9, 14, 19, 24, 29, 49]:

                
                img_train = utils.make_grid(
                    torch.cat((
                        input_mb,#[:, :, :256, :464],
                        recon_mb,
                    ), dim=0), nrow=batch_size
                )
                utils.save_image(
                    img_train,
                    f"torch_results/{args.exp}_img_train_{epoch + 1}.png"
                )
                model.eval()
                input_test_mb, recon_test_mb,  opt_out = eval(model=model,
                    test_loader=test_dataloader,
                    device=device)
                    
                model.train()
                img_test = utils.make_grid(
                    torch.cat((
                        input_test_mb,#[:, :, :256, :464],
                        recon_test_mb),
                        dim=0),
                        nrow=batch_size_test
                )
                utils.save_image(
                    img_test,
                    f"torch_results/{args.exp}_img_test_{epoch + 1}.png"
                
                )
                extract_mu_values(model, train_dataset_tensor, "cpu",
                        epoch + 1)
                model.to(device)
    mu_values, _ = extract_mu_values(model, train_dataset_tensor, "cpu", epoch)
    print(mu_values)
    
   

if __name__ == "__main__":
    args = parse_args()
    main(args)


