# -- Built-in modules -- #
import gc   
import os   
import sys  
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from tqdm import tqdm  # Progress bar
from sklearn.metrics import confusion_matrix
import pickle


scratch_dir = '/work3/s222651/'
data_dir = '/work3/s222651/data/data/'
savepatches_dir = '/work3/s222651/patches/'

#model_dir1 = '/work3/s222651/model1/70epochs_exp1/'
model_dir1 = '/work3/s222651/model1/exp2/'

model_path = os.path.join(model_dir1, 'best_model.pth')

# -- Environmental variables -- #
os.environ['AI4ARCTIC_DATA'] = '/work3/s222651/data/data/dataV2/'
os.environ['AI4ARCTIC_ENV'] = '/work3/s222651/'


from functions1 import chart_cbar, f1_metric, compute_metrics  # Functions to calculate metrics and show the relevant chart colorbar.
from torchdataloaders import Dataset, ValidationDataset, get_variable_options  #  dataloaders for training and validation.
from unet import UNet  # Convolutional Neural Network model
from utils1 import CHARTS, SIC_LOOKUP, SOD_LOOKUP, FLOE_LOOKUP, SCENE_VARIABLES, colour_str


### ---------------  TRAIN OPTIONS --------------- ### 

train_options = {
    # -- Training options -- #
    'path_to_processed_data': os.environ['AI4ARCTIC_DATA'],  
    'path_to_env': os.environ['AI4ARCTIC_ENV'], 
    #'validation_list': validation_files,
    'lr': 0.0001,  # Optimizer learning rate.
    'epochs': 70,  # Number of epochs before training stop.
    'epoch_len': 500,  # Number of batches for each epoch.
    'patch_size': 256,  # Size of patches sampled. Used for both Width and Height.
    'batch_size': 8,  # Number of patches for each batch.
    'loader_upsampling': 'nearest',  # How to upscale low resolution variables to high resolution.
    
    # -- Data prepraration lookups and metrics.
    'train_variables': SCENE_VARIABLES,  # Contains the relevant variables in the scenes.
    'charts': CHARTS,  # Charts to train on.
    'n_classes': {  # number of total classes in the reference charts, including the mask.
        'SIC': SIC_LOOKUP['n_classes'],
        'SOD': SOD_LOOKUP['n_classes'],
        'FLOE': FLOE_LOOKUP['n_classes']
    },
    'pixel_spacing': 80,  # SAR pixel spacing. 80 for the ready-to-train AI4Arctic Challenge dataset.
    'train_fill_value': 0,  # Mask value for SAR training data.
    'class_fill_values': {  # Mask value for class/reference data.
        'SIC': SIC_LOOKUP['mask'],
        'SOD': SOD_LOOKUP['mask'],
        'FLOE': FLOE_LOOKUP['mask'],
    },
    
    # -- Validation options -- #
    'chart_metric': {  # Metric functions for each ice parameter and the associated weight.
        #'SIC': {
        #    'func': r2_metric,
        #   'weight': 2,
        #},
        'SOD': {
            'func': f1_metric,
        #    'weight': 2,
        },
        #'FLOE': {
        #    'func': f1_metric,
        #    'weight': 1,
        
    },
    'num_val_scenes': 20,  # Number of scenes randomly sampled from train_list to use in validation.
    
    # -- GPU/cuda options -- #
    'gpu_id': 0,  # Index of GPU. In case of multiple GPUs.
    'num_workers': 8,  # Number of parallel processes to fetch data.
    'num_workers_val': 1,  # Number of parallel processes during validation.
    
    # -- U-Net Options -- #
    'unet_conv_filters': [16, 32, 64, 64],  # Number of filters in the U-Net.
    'conv_kernel_size': (3, 3),  # Size of convolutional kernels.
    'conv_stride_rate': (1, 1),  # Stride rate of convolutional kernels.
    'conv_dilation_rate': (1, 1),  # Dilation rate of convolutional kernels.
    'conv_padding': (1, 1),  # Number of padded pixels in convolutional layers.
    'conv_padding_style': 'zeros',  # Style of padding.
}
# Get options for variables, amsrenv grid, cropping and upsampling.
get_variable_options = get_variable_options(train_options)
test_files_directory = '/work3/s222651/test/data/testdata/'
test_files = [f for f in os.listdir(test_files_directory) if f.endswith('.nc')]
train_options['test_list'] = test_files


# To be used in test_upload.
#%store train_options  


##### ------------ LOAD TRAINING LIST ------------ #####


# load the files that are in the os.environ['AI4ARCTIC_DATA'] = '/work3/s222651/data/data/dataV2/' to the train_list
train_files_directory = '/work3/s222651/data/data/dataV2/'
train_files = [f for f in os.listdir(train_files_directory) if f.endswith('.nc')]
train_options['train_list'] = train_files

# same for the validation list as the training list
validation_files_directory = '/work3/s222651/validation/validationdata/'
validation_files = [f for f in os.listdir(validation_files_directory) if f.endswith('.nc')]
train_options['validation_dir'] = validation_files_directory
train_options['validation_files'] = validation_files


print('Options initialised')
print(f"training list: {train_options['train_list']}")
print(f"validation list: {train_options['validation_files']}")


#### --------- GPU SETUP --------- ####

if torch.cuda.is_available():
    print(colour_str('GPU available!', 'green'))
    print('Total number of available devices: ', colour_str(torch.cuda.device_count(), 'orange'))
    device = 'cuda'

else:
    print(colour_str('GPU not available.', 'red'))
    device = torch.device('cpu')




#### --------- DATALOADERS FOR PYTORCH --------- ####


# Custom dataset and dataloader.
dataset = Dataset(files=train_options['train_list'], options=train_options, save_dir = savepatches_dir)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=True, num_workers=8, pin_memory=True)
# - Setup of the validation dataset/dataloader. The same is used for model testing in 'test_upload.ipynb' with test = True (different masking)
dataset_val = ValidationDataset(options=train_options, files=train_options['validation_files'], test=False)
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=None, num_workers=8, shuffle=False)



######  --------  U-NET MODEL SETUP  --------  ######


# Setup U-Net model, adam optimizer, loss function 
net = UNet(options=train_options).to(device)
optimizer = torch.optim.Adam(list(net.parameters()), lr=train_options['lr'])
torch.backends.cudnn.benchmark = True  # Selects the kernel with the best performance for the GPU and given input size.

# Loss functions to use for each sea ice parameter.
# ignore_index argument discounts the masked values, ensuring that the model is not using these pixels to train on.
# It is equivalent to multiplying the loss of the relevant masked pixel with 0.
loss_functions = {chart: torch.nn.CrossEntropyLoss(ignore_index=train_options['class_fill_values'][chart]) \
                                                   for chart in train_options['charts']}



 ######## ------------  TRAINING LOOP  ------------ ########

best_F1_score = 0  # Best model score 
train_loss_history = []
val_loss_history = []

# -- Training Loop -- #
for epoch in tqdm(iterable=range(train_options['epochs']), position=0):
    gc.collect()  
    loss_sum = torch.tensor([0.])  # To sum the batch losses during the epoch
    net.train()  

    # Loop through the batches 
    for i, (batch_x, batch_y) in enumerate(tqdm(iterable=dataloader,total=train_options['epoch_len'])):
        torch.cuda.empty_cache()  # Empties the GPU cache freeing up memory.
        loss_batch = 0  # Reset from previous batch.
        
        #  Send data to GPU
        batch_x = batch_x.to(device, non_blocking=True)

        # mixed precision training to save memory
        with torch.cuda.amp.autocast():
            # - Forward pass
            output = net(batch_x)

            # Calculate loss
            for chart in train_options['charts']:
                loss_batch += loss_functions[chart](input=output[chart], target=batch_y[chart].to(device))

        # clear old gradients, not to accumulate 
        optimizer.zero_grad()

        # Backward pass to calculate the gradients
        loss_batch.backward()

        # Optimizer step
        optimizer.step()

        # sum the batch loss to have the epoch loss
        loss_sum += loss_batch.detach().item()

        # Average loss for monitoring the training 
        loss_epoch = torch.true_divide(loss_sum, i + 1).detach().item()
        print('\rMean training loss: ' + f'{loss_epoch:.3f}', end='\r')
        del output, batch_x, batch_y   # free up memory

    #del loss_sum
    loss_epoch = torch.true_divide(loss_sum, i + 1).detach().item()
    train_loss_history.append(loss_epoch)


    # -- Validation Loop -- #
    loss_batch = loss_batch.detach().item()  # For printing after the validation loop
    
    # Initilize the output and the reference pixels to calculate the scores after inference on all the scenes
    outputs_flat = {chart: np.array([]) for chart in train_options['charts']}
    inf_ys_flat = {chart: np.array([]) for chart in train_options['charts']}

    net.eval() 

    val_loss_sum = 0
    num_batches = 0
    

    # Loop through scenes 
    for inf_x, inf_y, masks, name in tqdm(iterable=dataloader_val,total=len(train_options['validation_files'])):
        torch.cuda.empty_cache()

        # ----------------  #

        # -  no gradients are calculated
        with torch.no_grad():
            inf_x = inf_x.to(device, non_blocking=True)
            output = net(inf_x)

        val_loss_batch = 0
        for chart in train_options['charts']:

            output_chart = output[chart]
            target_chart = inf_y[chart].unsqueeze(0).to(device).long()

            if torch.isinf(output_chart).any():
                print(f"inf values in output: {name}")
                print(f"output_chart values: {output_chart}")

            if torch.isinf(target_chart).any():
                print(f"inf values in target: {name}")
                print(f"target_chart values: {target_chart}")

            #print(f"output_chart min: {output_chart.min().item()}, output_chart max: {output_chart.max().item()}")

            try:
                val_loss = loss_functions[chart](input=output[chart], target=target_chart)

                if torch.isnan(val_loss).any():
                    print(f"nan values in loss: {name}")
                    raise ValueError(f"nan values in loss for file: {name}")
                
                val_loss_batch += val_loss

            except Exception as e:
                print(f"error computing loss for file {name}: {e}")
                raise

        if not torch.isnan(val_loss_batch).any():
            val_loss_sum += val_loss_batch.detach().item()
            num_batches += 1
        else:
            print(f"nan values in loss_batch: {name}")
            break

        # - Final output layer, and storing of non masked pixels.
        for chart in train_options['charts']:
            output[chart] = torch.argmax(output[chart], dim=1).squeeze().cpu().numpy()
            outputs_flat[chart] = np.append(outputs_flat[chart], output[chart][~masks[chart]])
            inf_ys_flat[chart] = np.append(inf_ys_flat[chart], inf_y[chart][~masks[chart]].numpy())

        print(val_loss_batch)
        val_loss_epoch = torch.true_divide(val_loss_sum, num_batches).detach().item()

        del inf_x, inf_y, masks, output  # free up memory

    print(num_batches)
    val_loss_history.append(val_loss_epoch)
  
    # compute the relevant scores (mainly SoD)
    chart_scores = {}
    for chart in train_options['charts']:
        scores = compute_metrics(true=inf_ys_flat[chart], pred=outputs_flat[chart])
        chart_scores[chart] = scores


    print("")
    print(f"Final batch loss: {loss_batch:.3f}")
    print(f"Epoch {epoch} score:")
    for chart in train_options['charts']:
        print(f"{chart} {train_options['chart_metric'][chart]['func'].__name__}: {chart_scores[chart]}%")

    if scores > best_F1_score:
        best_F1_score = scores
        torch.save(obj={'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_score': best_F1_score,
                        'scores': scores,
                        'epoch': epoch},
                        f=model_path)
    #del inf_ys_flat, outputs_flat  # free up memory

print("Validation loss history:", val_loss_history)

plt.figure(figsize=(10,5))
plt.plot(train_loss_history, label='Train loss',color='red',linestyle='-')
plt.plot(val_loss_history, label='Validation loss',color='blue',linestyle='--')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.title('Training and Validation loss on dataset')
plt.legend()
plt.savefig(model_dir1 + 'loss.png')
print(f"loss plot saved in {model_dir1} as loss.png")
#plt.show()
plt.close()