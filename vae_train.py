import os
import numpy as np
from torchvision import utils  
import torch

from datasets import *
from utils import *

torch.backends.cudnn.benchmark = True

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

    if args.mask:
        main = "masked_" + args.month  # Add 'masked_' before args.month
    else:
        main = args.month  # Keep args.month as it is

    main = main.lstrip("./")  # Remove the "./" at the beginning
    os.makedirs(main, exist_ok=True)

    directory =  os.path.join(main,f'./torch_results_zdim{args.z_dim}')

    #create directory if does not exist
    os.makedirs(directory, exist_ok=True)

    #Create sub directory for every beta tested
    sub_directory = os.path.join(directory,f"{args.beta}")
    #create directory if does not exist
    os.makedirs(sub_directory, exist_ok=True)
    
    out_dir =  os.path.join(main,f'./torch_logs_{args.month}_zdim{args.z_dim}')
    os.makedirs(out_dir, exist_ok=True)

    #Create sub directory for every beta tested
    sub_out_dir = os.path.join(out_dir,f"{args.beta}")
    os.makedirs(sub_out_dir, exist_ok=True)



    model.eval()
    model.to(device_extraction)
   
    mu_values, _ = model.encoder(train_dataset_tensor)    
    

    # Convert the batch to a NumPy array
    mu_values_np = mu_values.detach().cpu().numpy()
    print(f"The mu shape is {mu_values_np.shape}")

    # Export to csv
    file_path = os.path.join(sub_directory, f'mu_values_{epoch}.csv')

    np.savetxt(file_path, mu_values_np, delimiter=',')
    print(f"Mu values saved to: {file_path}")

    return mu_values, mu_values_np 

def extract_mu_values_early_stopping(model, train_dataset_tensor, device_extraction="cpu"):
    model.eval()
    model.to(device_extraction)

    mu_values,_ = model.encoder(train_dataset_tensor)    

    # Convert the batch to a NumPy array
    mu_values_np = mu_values.detach().cpu().numpy()


    return mu_values_np

   



def main(args):

    best_loss = float('inf')  # Initialize best loss to positive infinity
    patience = 10  # Define the number of epochs to wait for loss improvement
    num_epochs_without_improvement = 0
    saved_mu_values = None


    # define directory outside main function so it be used elsewhere

        
    if args.mask:
        main = "masked_" + args.month  # Add 'masked_' before args.month
    else:
        main = args.month  # Keep args.month as it is

    main = main.lstrip("./")  # Remove the "./" at the beginning
    os.makedirs(main, exist_ok=True)

    directory =  os.path.join(main,f'./torch_results_zdim{args.z_dim}')

    #create directory if does not exist
    os.makedirs(directory, exist_ok=True)

    #Create sub directory for every beta tested

    sub_directory = os.path.join(directory,f"{args.beta}")
    #create directory if does not exist
    os.makedirs(sub_directory, exist_ok=True)
    
    out_dir =  os.path.join(main,f'./torch_logs_{args.month}_zdim{args.z_dim}')
    os.makedirs(out_dir, exist_ok=True)

    #Create sub directory for every beta tested
    sub_out_dir = os.path.join(out_dir,f"{args.beta}")
    os.makedirs(sub_out_dir, exist_ok=True)
    
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
    beta = args.beta
    img_size = args.img_size
    batch_size = args.batch_size
    batch_size_test = args.batch_size_test
    month = args.month

    print("Nb channels", nb_channels, "img_size", img_size, 
        "mini batch size", batch_size, "beta", beta,"month", month)


    if not os.path.isdir(sub_out_dir):
        os.mkdir(sub_out_dir)
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
            f_name = os.path.join(sub_out_dir, f"{args.exp}_loss_values_{args.z_dim}_{args.beta}.txt")
            print_loss_logs(f_name, sub_out_dir, loss_dict, epoch, args.exp)

            # Check if the loss improved
            if abs(loss) < abs(best_loss):
                best_loss = loss
                best_loss_dict = loss_dict
                best_epochs = epoch
                num_epochs_without_improvement = 0
                # Save the current reconstruction and mu values
                saved_reconstruction = recon_mb
                saved_mu_values = extract_mu_values_early_stopping(model, train_dataset_tensor, "cpu")
            else:
                num_epochs_without_improvement += 1

            # Check if early stopping criteria are met
            if num_epochs_without_improvement >= patience:
                print("Early stopping: Loss did not improve for {} epochs.".format(patience))
                print(f"Best loss achieved at epoch {best_epochs}, Loss: {best_loss}")
                break
            
            model.to(device)

                 
            
        
            
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
                    os.path.join(sub_directory, f"{args.exp}_img_train_{epoch + 1}.png")
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
                    os.path.join(sub_directory, f"{args.exp}_img_test_{epoch + 1}.png")
              
                )
                extract_mu_values(model, train_dataset_tensor, "cpu",
                        epoch + 1)
                model.to(device)

       
    # Save the mu values at the end of the training process if early stopping was triggered
    if saved_mu_values is not None and num_epochs_without_improvement >= patience:
        model.to("cpu")
        file_path = os.path.join(sub_directory, f'mu_values_final_{best_epochs}.csv')
        np.savetxt(file_path, saved_mu_values, delimiter=',')
        print(f"Final Mu values saved to: {file_path}")

        # Write loss dict
        f_name = os.path.join(sub_out_dir, f"{args.exp}_final_loss_values_{args.z_dim}_{args.beta}.txt")
        print_loss_logs(f_name, sub_out_dir, best_loss_dict, epoch, args.exp)


    if saved_reconstruction is not None and num_epochs_without_improvement >= patience:
            img_train = utils.make_grid(
                torch.cat((
                    input_mb,  # [:, :, :256, :464],
                    saved_reconstruction,
                ), dim=0), nrow=batch_size
            )
            utils.save_image(
                img_train,
                os.path.join(sub_directory, f"{args.exp}_final_img_train_{best_epochs}_.png")
            )
            img_test = utils.make_grid(
                    torch.cat((
                        input_test_mb,#[:, :, :256, :464],
                        recon_test_mb),
                        dim=0),
                        nrow=batch_size_test
                )
            utils.save_image(
                    img_test,
                    os.path.join(sub_directory, f"{args.exp}_final_img_test_{best_epochs}.png")
              
                )

    mu_values, _ = extract_mu_values(model, train_dataset_tensor, "cpu", epoch)
    # print(mu_values)


if __name__ == "__main__":
    args = parse_args()
    main(args)



