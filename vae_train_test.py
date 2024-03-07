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
                   get_Yield_Dataset,
                   load_model_parameters,
                   load_vqvae,
                   load_resnet_vae,
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



############################ TESTE BRANCH CLASSICIFATION

# def train(model, train_loader,Yield_train_dataloader, device, optimizer, epoch, num_inner_epochs=100):
#     model.train()
#     train_loss = 0
#     loss_dict = {}

#     for batch_idx, (input_mb) in enumerate(train_loader):
#         print(batch_idx + 1, end=", ", flush=True)
#         input_mb = input_mb.to(device)
#         optimizer.zero_grad()  # otherwise grads accumulate in backward

#         loss, recon_mb, loss_dict_new = model.step(input_mb)

#         # Inner LOOCV loop for classification
#         model.freeze()

#         y_true_list = []
#         y_pred_list = []

#         # Assuming mu_values is defined somewhere as a tensor of mu values
#         mu_values = model.mu_values

#         with torch.no_grad():
            
#             for exclude_index in enumerate(Yield_train_dataloader):
#                 dataset = Yield_train_dataloader
#                 data_point = dataset[exclude_index]
#                 input_data = mu_values.to(device)
#                 y_true = data_point
#                 y_true_list.append(y_true)

#                 # Train the VAE model on the training set with one data point excluded
#                 excluded_data_indices = list(range(len(dataset)))
#                 excluded_data_indices.remove(exclude_index)
#                 train_dataset = torch.utils.data.Subset(dataset, excluded_data_indices)
#                 train_loader = Yield_train_dataloader

#                 # Create a new optimizer for each inner LOOCV iteration
#                 inner_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#                 # Inner loop for training the model
#                 for inner_epoch in range(num_inner_epochs):
#                     model.classification()
#                     for input_mb_inner in train_loader:
#                         input_mb_inner = input_mb_inner.to(device)
#                         inner_optimizer.zero_grad()

#                         loss_inner, _, _ = model.step(input_mb_inner)
#                         loss_inner.backward()
#                         inner_optimizer.step()

#                 # Get the classification result (y_hat)
#                 y_pred = model.classification(input_data)
#                 y_pred_list.append(y_pred.item())

#         # Compute the Brier score using the true labels and predicted probabilities
#         BSS = model.brier_skill_score(y_true_list, y_pred_list)

#         # Append the BSS slot to the loss_dict and assign the value
#         loss_dict["BSS"] = BSS
#         loss_dict_new["BSS"] = BSS
#         loss = loss - BSS
#         (-loss).backward()
#         train_loss += loss.item()
#         loss_dict = update_loss_dict(loss_dict, loss_dict_new)
#         optimizer.step()

#     nb_mb_it = (len(train_loader.dataset) // input_mb.shape[0])
#     train_loss /= nb_mb_it
#     loss_dict = {k: v / nb_mb_it for k, v in loss_dict.items()}

    # return train_loss, input_mb, recon_mb, loss_dict



def main(args):
    # define directory outside main function so it be used elsewhere

    directory = f'./torch_results_{args.month}_zdim{args.z_dim}'
    #create directory if does not exist
    os.makedirs(directory, exist_ok=True)

    out_dir = f'./torch_logs_{args.month}_zdim{args.z_dim}'
    os.makedirs(out_dir, exist_ok=True)
    
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
    
    #Classification

    # Yield_train_dataloader, Yield_train_dataset = get_Yield_Dataset(args)



    train_dataset_tensor = []
    for i in range(len(train_dataset)):
        train_dataset_tensor.append(torch.from_numpy(train_dataset[i]))
    train_dataset_tensor = torch.stack(train_dataset_tensor,axis = 0)
    nb_channels = args.nb_channels
    beta = args.beta
    img_size = args.img_size
    batch_size = args.batch_size
    batch_size_test = args.batch_size_test

    print("Nb channels", nb_channels, "img_size", img_size, 
        "mini batch size", batch_size, "beta", beta)


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
                    os.path.join(directory, f"{args.exp}_img_train_{epoch + 1}.png")
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
                    os.path.join(directory, f"{args.exp}_img_test_{epoch + 1}.png")
              
                )
                extract_mu_values(model, train_dataset_tensor, "cpu",
                        epoch + 1)
                model.to(device)
    mu_values, _ = extract_mu_values(model, train_dataset_tensor, "cpu", epoch)
    print(mu_values)

# Plot the latent space
    plot_latent(model,train_dataset_tensor,device)
   

if __name__ == "__main__":
    args = parse_args()
    main(args)


