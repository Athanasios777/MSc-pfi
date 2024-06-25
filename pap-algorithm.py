

import os

import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score
from copy import deepcopy
import random

from functions1 import chart_cbar, f1_metric, compute_metrics
from torchdataloaders import TestDatasetPermute,get_variable_options
from unet import UNet
from utils1 import CHARTS, SIC_LOOKUP, SOD_LOOKUP, FLOE_LOOKUP, SCENE_VARIABLES, colour_str



scratch_dir = '/work3/s222651/'
data_dir = '/work3/s222651/data/data/'
savepatches_dir = '/work3/s222651/patches/'

model_dir1 = '/work3/s222651/model1/70epochs_16channelsNEWNEW/'
model_path = os.path.join(model_dir1, 'best_model.pth')


# -- Environmental variables -- #
os.environ['AI4ARCTIC_DATA'] = '/work3/s222651/data/data/dataV2/'
os.environ['AI4ARCTIC_ENV'] = '/work3/s222651/'



train_options = {
    # -- Training options -- #
    'path_to_processed_data': os.environ['AI4ARCTIC_DATA'],  
    'path_to_env': os.environ['AI4ARCTIC_ENV'], 
    #'train_files': 'C:\\Users\\HpRyzen7\\Desktop\\Thesis\\AI4Arctic\\shaptest\\train\\',
    'test_files': '/work3/s222651/test/data/testdata/', 
    #'path_to_env': os.environ['AI4ARCTIC_ENV'],  
    'lr': 0.0001,  # Optimizer learning rate.
    'epochs': 10,  # Number of epochs before training stop.
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




if torch.cuda.is_available():
    print(colour_str('GPU available!', 'green'))
    print('Total number of available devices: ', colour_str(torch.cuda.device_count(), 'orange'))
    device = 'cuda'
else:
    print(colour_str('GPU not available.', 'red'))
    device = torch.device('cpu')
print('Loading model...')

# Setup U-Net model, adam optimizer, loss function and dataloader.
net = UNet(options=train_options).to(device)
model = '/work3/s222651/model1/70epochs_16channelsNEWNEW/best_model.pth'
checkpoint = torch.load(model, map_location=device)
net.load_state_dict(checkpoint['model_state_dict'])

print('Model successfully loaded.')


test_files_directory = '/work3/s222651/test/data/testpap/'
test_files = [f for f in os.listdir(test_files_directory) if f.endswith('.nc') and not f.endswith('_reference.nc')]
train_options['test_list'] = test_files

dataset_test = TestDatasetPermute(options=train_options, files=train_options['test_list'],test_dir=test_files_directory)
asid = torch.utils.data.DataLoader(dataset_test, batch_size=None, num_workers=train_options['num_workers_val'], shuffle=False)
print('Setup ready')




def evaluate_model_f1(model, data_loader, device):
    """
    F1 score of the model on test (unseen) data
    """
    model.eval()
    true_labels = []
    predictions = []
    with torch.no_grad():
        for batch_x, batch_y, masks, _ in data_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)

            for chart in masks:
                mask = masks[chart].view(-1)
                predicted_labels = torch.argmax(output['SOD'], dim=1).view(-1)
                predicted_labels = predicted_labels[mask].cpu().numpy()
                true_labels.extend(batch_y[chart].view(-1)[mask].numpy())
                predictions.extend(predicted_labels)

    return f1_score(true_labels, predictions, average='weighted')

baseline_f1 = evaluate_model_f1(net, asid, device)
# print the baseline F1 score with 2 decimal digits
print(f'Baseline F1 score: {baseline_f1:.2f}')


# Permute and Predict  (PaP) algorithm implementation

def permute_and_predict(model, dataloader, device, seed=42):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model.eval()

    all_F1_scores = []
    importance_scores = []


    num_channels = None  

    with torch.no_grad():
        
        for inputs, _, _, _ in dataloader:
            if num_channels is None:
                num_channels = inputs.size(1)  # Determine the number of channels from the first batch
            break  # Break after the first batch to just read the number of channels

        for channel in range(2, num_channels):  #  permutation from the third channel (PMW channels only)
            all_true_labels = []
            all_predictions = []

            # Process each batch
            for inputs, labels, masks, _ in dataloader:
                permuted_inputs = inputs.clone()

                # Permute the current channel for the entire batch
                for i in range(inputs.size(0)):
                    original = permuted_inputs[i, channel, :, :].reshape(-1).clone()   # Flatten the 2D data
                    permuted_data = original[torch.randperm(original.numel())].reshape(inputs.size(2), inputs.size(3))  # Shuffle the data
                    permuted_inputs[i, channel, :, :] = permuted_data  #  shuffled data back

                permuted_inputs = permuted_inputs.to(device)
                outputs = model(permuted_inputs)
                predicted_labels = torch.argmax(outputs['SOD'], dim=1)

                # Collect all true labels and predictions
                for chart in masks:
                    valid_mask = masks[chart].view(-1)
                    true_labels_ext = labels[chart].view(-1)[valid_mask].cpu().numpy()
                    predictions_ext = predicted_labels.view(-1)[valid_mask].cpu().numpy()
                    all_true_labels.extend(true_labels_ext)
                    all_predictions.extend(predictions_ext)

            # Calculate F1 score for the whole dataset with the current channel permuted
            permuted_f1 = f1_score(all_true_labels, all_predictions, average='weighted')
            all_F1_scores.append(permuted_f1)

            # Calculate the importance score
            importance_score = baseline_f1 - permuted_f1
            importance_scores.append(importance_score)

    return all_F1_scores, importance_scores


f1_scores, importance_scores = permute_and_predict(net, asid, device)



output_dir = 'work3/s222651/pap/'
os.makedirs(output_dir, exist_ok=True)


file_path = os.path.join(output_dir, 'importance_scores_datasetmodel3seed42.json')
with open(file_path, "w") as f:
    json.dump(importance_scores, f)

print("Importance scores saved to:", file_path)


file_path1 = os.path.join(output_dir, 'f1_scores_datasetmodel3seed42.json')
with open(file_path1, "w") as f:
    json.dump(f1_scores, f) 

print("F1 scores saved to:", file_path1)