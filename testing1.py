
import os

import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from sklearn.metrics import confusion_matrix
from tqdm.notebook import tqdm

from functions1 import chart_cbar, f1_metric, compute_metrics
from torchdataloaders import TestDataset,get_variable_options, TestDatasetPermute
from unet import UNet
from utils1 import CHARTS, SIC_LOOKUP, SOD_LOOKUP, FLOE_LOOKUP, SCENE_VARIABLES, colour_str

os.environ['AI4ARCTIC_DATA'] = '/work3/s222651/data/data/dataV2/'
os.environ['AI4ARCTIC_ENV'] = '/work3/s222651/'

scratch_dir = '/work3/s222651/'
data_dir = '/work3/s222651/data/data/'
savetestpatches_dir = '/work3/s222651/test/patches/'


model_dir1 = '/work3/s222651/model1/'

#%store -r train_options
train_options = {
    # -- Training options -- #
    'path_to_processed_data': os.environ['AI4ARCTIC_DATA'], 
    'test_files': '/work3/s222651/test/data/testdata/', 
    'path_to_env': os.environ['AI4ARCTIC_ENV'],  
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


if torch.cuda.is_available():
    print(colour_str('GPU available!', 'green'))
    print('Total number of available devices: ', colour_str(torch.cuda.device_count(), 'orange'))
    device = 'cuda'

else:
    print(colour_str('GPU not available.', 'red'))
    device = torch.device('cpu')


print('Loading model.')
net = UNet(options=train_options).to(device)


model = '/work3/s222651/model1/exp2/best_model.pth'
checkpoint = torch.load(model, map_location=device)
net.load_state_dict(checkpoint['model_state_dict'])


print('Model successfully loaded.')


test_files_directory = '/work3/s222651/test/data/testpap/'
test_files = [f for f in os.listdir(test_files_directory) if f.endswith('.nc') and not f.endswith('_reference.nc')]
train_options['test_list'] = test_files


# normal testdataset for the creation of the inference images.
dataset_test = TestDatasetPermute(options=train_options, files=train_options['test_list'], test_dir = test_files_directory)
asid = torch.utils.data.DataLoader(dataset_test, batch_size=None, num_workers=train_options['num_workers_val'], shuffle=False)
print('Setup ready')

# # torch dataloader for the test dataset along with the labels
# dataset_test = TestDatasetPermute(options=train_options, files=train_options['test_list'],test_dir=test_files_directory)
# asid = torch.utils.data.DataLoader(dataset_test, batch_size=None, num_workers=train_options['num_workers_val'], shuffle=False)
# print('Setup ready')


all_true_labels = []
all_predictions = []


os.makedirs('inference/exp1', exist_ok=True)
net.eval()
for inf_x, inf_y, masks, scene_name in tqdm(iterable=asid, total=len(train_options['test_list']), colour='green', position=0):
    scene_name = scene_name
    torch.cuda.empty_cache()
    inf_x = inf_x.to(device, non_blocking=True)

    print(inf_x.shape)  # [batch_size, channels input, H, W]


    with torch.no_grad(), torch.cuda.amp.autocast():
        pred = net(inf_x) 
        print(pred['SOD'].shape)      # [batch_size, 7, H, W]

    logits = pred['SOD'].cpu().numpy()


    for chart in train_options['charts']:
        pred[chart] = torch.argmax(pred[chart], dim=1).squeeze().cpu().numpy()
        print(f"Shape of masks[chart] before conversion: {masks[chart].shape}")

        masks[chart] = masks[chart].cpu().numpy()


        flat_mask = masks[chart].ravel()
        flat_pred = pred[chart].ravel()
        flat_true_labels = inf_y[chart].cpu().numpy().astype(int).ravel()


        # for visualization
        display_pred = pred[chart].astype(float)
        display_pred[~masks[chart]] = np.nan

        # confusion matrix
        masked_pred = flat_pred[flat_mask]
        masked_true_labels = flat_true_labels[flat_mask]

        all_true_labels.extend(masked_true_labels)
        all_predictions.extend(masked_pred)


    # - Show the scene inference 
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))


    ax.imshow(display_pred, vmin=0, vmax=train_options['n_classes'][chart] - 2, cmap='jet', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    chart_cbar(ax=ax, n_classes=train_options['n_classes'][chart], chart=chart, cmap='jet')

    preds_and_masks_dict = {f"{chart}_pred": pred[chart] for chart in train_options['charts']}
    
    numpy_masks = {}

    #preds_and_masks_dict.update({f"{chart}_mask": masks[chart].cpu().numpy() for chart in train_options['charts']})
    for chart,mask in masks.items():
        if isinstance(mask, torch.Tensor):
            numpy_masks[chart] = mask.cpu().numpy()
        else:
            numpy_masks[chart] = mask
    
    preds_and_masks_dict["test_mask"] = numpy_masks

    # Save the dictionary of numpy arrays to a compressed npz file
    np.savez_compressed(f"inference/{scene_name}_data.npz", **preds_and_masks_dict)
    
    #plt.suptitle(f"Scene: {scene_name}", y=0.65)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.5, hspace=-0)
    fig.savefig(f"inference/{scene_name}.png", format='png', dpi=128, bbox_inches="tight")
    plt.close('all')



cm = confusion_matrix(all_true_labels,all_predictions)

# save the confusion matrix
cm_path = f"/work3/s222651/model1/exp2/confusion_matrix.npy"
np.save(cm_path, cm)
print(f"Confusion matrix saved to {cm_path}")


