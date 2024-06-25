

# Dataloaders for the ASID v2 ready-to-train challenge dataset, inspired by the ASID v2 dataloaders.


# -- Built-in modules -- #
import os

# -- Third-party modules -- #
import copy
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


class Dataset(Dataset):
    """Pytorch dataset for loading batches of patches of scenes from the ASID V2 data set."""

    def __init__(self, options, files,save_dir):
        self.options = options
        self.files = files
        self.save_dir = save_dir
        self.file_ids = {file: i for i, file in enumerate(files)}


        # Channel numbers in patches, includes reference channel.
        self.patch_c = len(self.options['train_variables']) + len(self.options['charts'])

    def __len__(self):
 
        return self.options['epoch_len']
    
    
    def random_crop(self, scene):
        """
        random cropping in scene.

        Params
        ----------
        scene :
            Xarray dataset; a scene from ASID v2 ready-to-train challenge dataset.

        Return
        -------
        patch :
            Numpy array with shape (len(train_variables), patch_height, patch_width). None if empty patch
                                          16,                256,            256
        """
        patch = np.zeros((len(self.options['full_variables']) + len(self.options['amsrenv_variables']),
                          self.options['patch_size'], self.options['patch_size']))
        
        # Get random index to crop from.
        row_rd = np.random.randint(low=0, high=scene['SOD'].values.shape[0] - self.options['patch_size'])
        col_rd = np.random.randint(low=0, high=scene['SOD'].values.shape[1] - self.options['patch_size'])
        # Equivalent in amsr and env variable grid.
        amsrenv_row = row_rd / self.options['amsrenv_delta']
        amsrenv_row_dec = int(amsrenv_row - int(amsrenv_row))  # Used in determining the location of the crop in between pixels.
        amsrenv_row_index_crop = amsrenv_row_dec * self.options['amsrenv_delta'] * amsrenv_row_dec
        amsrenv_col = col_rd / self.options['amsrenv_delta']
        amsrenv_col_dec = int(amsrenv_col - int(amsrenv_col))
        amsrenv_col_index_crop = amsrenv_col_dec * self.options['amsrenv_delta'] * amsrenv_col_dec
        
        # - Discard patches with too many meaningless pixels
        if np.sum(scene['SOD'].values[row_rd: row_rd + self.options['patch_size'], 
                                      col_rd: col_rd + self.options['patch_size']] != self.options['class_fill_values']['SOD']) > 1:
            
            # Crop full resolution variables.
            patch[0:len(self.options['full_variables']), :, :] = scene[self.options['full_variables']].isel(
                sar_lines=range(row_rd, row_rd + self.options['patch_size']),
                sar_samples=range(col_rd, col_rd + self.options['patch_size'])).to_array().values
            # Crop and upsample low resolution variables.
            patch[len(self.options['full_variables']):, :, :] = torch.nn.functional.interpolate(
                input=torch.from_numpy(scene[self.options['amsrenv_variables']].to_array().values[
                    :, 
                    int(amsrenv_row): int(amsrenv_row + np.ceil(self.options['amsrenv_patch'])),
                    int(amsrenv_col): int(amsrenv_col + np.ceil(self.options['amsrenv_patch']))]
                ).unsqueeze(0),
                size=self.options['amsrenv_upsample_shape'],
                mode=self.options['loader_upsampling']).squeeze(0)[
                :,
                int(np.around(amsrenv_row_index_crop)): int(np.around(amsrenv_row_index_crop + self.options['patch_size'])),
                int(np.around(amsrenv_col_index_crop)): int(np.around(amsrenv_col_index_crop + self.options['patch_size']))].numpy()

        # If not any valid pixels - return None.
        else:
            patch = None

        return patch


    def prep_dataset(self, patches):
        """
        Convert patches from 4D numpy array to 4D torch tensor for loading them in training.

        Parameters
        ----------
        patches : ndarray
            Patches sampled from ASID v2 ready-to-train challenge dataset scenes [PATCH, CHANNEL, H, W].

        Returns
        -------
        x :
            4D torch tensor; ready training data.
        y : Dict
            Dictionary with 3D torch tensors for each chart; reference data for training data x.
        """
        # Convert training data to tensor.
        x = torch.from_numpy(patches[:, len(self.options['charts']):]).type(torch.float)

        # Store charts in y dictionary.
        y = {}
        for idx, chart in enumerate(self.options['charts']):
            y[chart] = torch.from_numpy(patches[:, idx]).type(torch.long)

        return x, y


    def __getitem__(self, idx):
        """
        Get batch. (required by Pytorch implementation)

        Returns
        -------
        x :
            4D torch tensor; ready training data.
        y : Dict
            Dictionary with 3D torch tensors for each chart; reference data for training data x.
        """
        # Placeholder to fill with data.
        patches = np.zeros((self.options['batch_size'], self.patch_c,
                            self.options['patch_size'], self.options['patch_size']))
        sample_n = 0

        # Continue until batch is full.
        while sample_n < self.options['batch_size']:
            # - Open memory location of scene. Uses 'Lazy Loading'.
            scene_id = np.random.randint(low=0, high=len(self.files), size=1).item()

            # - Load scene
            scene = xr.open_dataset(os.path.join(self.options['path_to_processed_data'], self.files[scene_id]))
            file_id = self.file_ids[self.files[scene_id]]
            # - Extract patches
            try:
                scene_patch = self.random_crop(scene)
            except:
                print(f"Cropping in {self.files[scene_id]} failed.")
                print(f"Scene size: {scene['SOD'].values.shape} for crop shape: ({self.options['patch_size']}, {self.options['patch_size']})")
                print('Skipping scene.')
                continue
            
            if scene_patch is not None:

                # file_name  = f"patch_file{file_id}_batch{idx}_sample{sample_n}.pt"
                # scene_patch_tens = torch.from_numpy(scene_patch).float()
                # file_path = os.path.join(self.save_dir, file_name)
                # torch.save(torch.from_numpy(scene_patch_tens), file_path)
                # -- Stack the scene patches in patches
                patches[sample_n, :, :, :] = scene_patch
                sample_n += 1 # Update the index.

        # Prepare training arrays
        x, y = self.prep_dataset(patches=patches)

        return x, y
    

   #######---------------------------#######################


class ValidationDataset(Dataset):
    """Pytorch dataset for loading full scenes from the ASID  v2 ready-to-train challenge dataset for validation."""

    def __init__(self, options, files, test=False):
        self.options = options
        self.files = files
        self.test = test

    def __len__(self):

        return len(self.options['validation_files'])

    def prep_scene(self, scene):
        """
        Upsample low resolution to match charts and SAR resolution. Convert patches from 4D numpy array to 4D torch tensor.

        Parameters
        ----------
        scene :

        Returns
        -------
        x :
            4D torch tensor, ready training data.
        y :
            Dict with 3D torch tensors for each reference chart; reference inference data for x. None if test is true.
        """
        x = torch.cat((torch.from_numpy(scene[self.options['sar_variables']].to_array().values).unsqueeze(0),
                      torch.nn.functional.interpolate(
                          input=torch.from_numpy(scene[self.options['amsrenv_variables']].to_array().values).unsqueeze(0),
                          size=scene['nersc_sar_primary'].values.shape, 
                          mode=self.options['loader_upsampling'])),
                      axis=1)
        
        if not self.test:
            y = {chart: scene[chart].values for chart in self.options['charts']}

        else:
            y = None
        
        return x, y

    def __getitem__(self, idx):
        """
        Get scene. Function required by Pytorch dataset.

        Returns
        -------
        x :s
            4D torch tensor; ready inference data.
        y :
            Dict with 3D torch tensors for each reference chart; reference inference data for x. None if test is true.
        masks :
            Dict with 2D torch tensors; mask for each chart for loss calculation. Contain only SAR mask if test is true.
        name : str
            Name of scene.

        """
        file_path = os.path.join(self.options['validation_dir'], self.options['validation_files'][idx])
        scene = xr.open_dataset(file_path)

        x, y = self.prep_scene(scene)
        name = self.options['validation_files'][idx]
        
        if not self.test:
            masks = {}
            for chart in self.options['charts']:
                masks[chart] = (y[chart] == self.options['class_fill_values'][chart]).squeeze()
                
        else:
            masks = (x.squeeze()[0, :, :] == self.options['train_fill_value']).squeeze()

        return x, y, masks, name




################# -------------------------- ###################################




class TestDataset(Dataset):
    """Pytorch dataset for loading full scenes from the ASID ready-to-train challenge dataset for inference."""

    def __init__(self, options, files, test=False):
        self.options = options
        self.files = files
        self.test = test

    def __len__(self):

        return len(self.files)

    def prep_scene(self, scene):
        """
        Upsample low resolution to match charts and SAR resolution. Convert patches from 4D numpy array to 4D torch tensor.

        Parameters
        ----------
        scene :

        Returns
        -------
        x :
            4D torch tensor, ready training data.
        y :
            Dict with 3D torch tensors for each reference chart; reference inference data for x. None if test is true.
        """
        x = torch.cat((torch.from_numpy(scene[self.options['sar_variables']].to_array().values).unsqueeze(0),
                      torch.nn.functional.interpolate(
                          input=torch.from_numpy(scene[self.options['amsrenv_variables']].to_array().values).unsqueeze(0),
                          size=scene['nersc_sar_primary'].values.shape, 
                          mode=self.options['loader_upsampling'])),
                      axis=1)
        
        if not self.test:
            y = {chart: scene[chart].values for chart in self.options['charts']}

        else:
            y = None
        
        return x, y

    def __getitem__(self, idx):
        """
        Get scene. Function required by Pytorch dataset.

        Returns
        -------
        x :s
            4D torch tensor; ready inference data.
        y :
            Dict with 3D torch tensors for each reference chart; reference inference data for x. None if test is true.
        masks :
            Dict with 2D torch tensors; mask for each chart for loss calculation. Contain only SAR mask if test is true.
        name : str
            Name of scene.

        """
        scene = xr.open_dataset(os.path.join(self.options['test_files'], self.files[idx]))

        x, y = self.prep_scene(scene)
        name = self.files[idx]
        
        if not self.test:
            masks = {}
            for chart in self.options['charts']:
                masks[chart] = (y[chart] == self.options['class_fill_values'][chart]).squeeze()
                
        else:
            masks = (x.squeeze()[0, :, :] == self.options['train_fill_value']).squeeze()

        return x, y, masks, name
    

###################### ------------------------------------- ##################################

def get_variable_options(train_options: dict):
    """
    Get amsr and env grid options, crop shape and upsampling shape.

    Parameters
    ----------
    train_options: dict
        Dictionary with training options.
    
    Returns
    -------
    train_options: dict
        Updated with amsrenv options.
    """
    train_options['amsrenv_delta'] = 50 / (train_options['pixel_spacing'] // 40)
    train_options['amsrenv_patch'] = train_options['patch_size'] / train_options['amsrenv_delta']
    train_options['amsrenv_patch_dec'] = int(train_options['amsrenv_patch'] - int(train_options['amsrenv_patch']))
    train_options['amsrenv_upsample_shape'] = (int(train_options['patch_size'] + \
                                                   train_options['amsrenv_patch_dec'] * \
                                                   train_options['amsrenv_delta']),
                                               int(train_options['patch_size'] +  \
                                                   train_options['amsrenv_patch_dec'] * \
                                                   train_options['amsrenv_delta']))
    train_options['sar_variables'] = [variable for variable in train_options['train_variables'] \
                                      if 'sar' in variable or 'map' in variable]
    train_options['full_variables'] = np.hstack((train_options['charts'], train_options['sar_variables']))
    train_options['full_variables1'] = train_options['sar_variables']

    train_options['amsrenv_variables'] = [variable for variable in train_options['train_variables'] \
                                          if 'sar' not in variable and 'map' not in variable]
    
    return train_options


###################### ------------------------------------- ##################################



class TestDatasetPatches(Dataset):
    def __init__(self,options,files, save_dir):
        self.options = options
        self.save_dir = save_dir
        self.file_ids = {file: i for i, file in enumerate(files)}
        self.files = files
        self.patch_c = len(self.options['train_variables'])

    def __len__(self):
        return len(self.files)
    
    def random_crop(self, scene):

        patch = np.zeros((len(self.options['full_variables1']) + len(self.options['amsrenv_variables']),
                          self.options['patch_size'], self.options['patch_size']))
        
        # Get random index to crop from.
        row_rd = np.random.randint(low=0, high=scene['nersc_sar_primary'].values.shape[0] - self.options['patch_size'])
        col_rd = np.random.randint(low=0, high=scene['nersc_sar_primary'].values.shape[1] - self.options['patch_size'])
        # Equivalent in amsr and env variable grid.
        amsrenv_row = row_rd / self.options['amsrenv_delta']
        amsrenv_row_dec = int(amsrenv_row - int(amsrenv_row))  # Used in determining the location of the crop in between pixels.
        amsrenv_row_index_crop = amsrenv_row_dec * self.options['amsrenv_delta'] * amsrenv_row_dec
        amsrenv_col = col_rd / self.options['amsrenv_delta']
        amsrenv_col_dec = int(amsrenv_col - int(amsrenv_col))
        amsrenv_col_index_crop = amsrenv_col_dec * self.options['amsrenv_delta'] * amsrenv_col_dec


        # Crop full resolution variables.
        patch[0:len(self.options['full_variables1']), :, :] = scene[self.options['full_variables1']].isel(
            sar_lines=range(row_rd, row_rd + self.options['patch_size']),
            sar_samples=range(col_rd, col_rd + self.options['patch_size'])).to_array().values
         # Crop and upsample low resolution variables.
        patch[len(self.options['full_variables1']):, :, :] = torch.nn.functional.interpolate(
            input=torch.from_numpy(scene[self.options['amsrenv_variables']].to_array().values[
                :, 
                int(amsrenv_row): int(amsrenv_row + np.ceil(self.options['amsrenv_patch'])),
                int(amsrenv_col): int(amsrenv_col + np.ceil(self.options['amsrenv_patch']))]
            ).unsqueeze(0),
            size=self.options['amsrenv_upsample_shape'],
            mode=self.options['loader_upsampling']).squeeze(0)[
            :,
            int(np.around(amsrenv_row_index_crop)): int(np.around(amsrenv_row_index_crop + self.options['patch_size'])),
            int(np.around(amsrenv_col_index_crop)): int(np.around(amsrenv_col_index_crop + self.options['patch_size']))].numpy()

        return patch



    def prep_dataset(self, patches):
        # Convert training data to tensor
        x = torch.from_numpy(patches).type(torch.float)

        return x
    
    def __getitem__(self, idx):
        patches = np.zeros((self.options['batch_size'], self.patch_c,
                            self.options['patch_size'], self.options['patch_size']))
        sample_n = 0

        # Continue until batch is full.
        while sample_n < self.options['batch_size']:
            # - Open memory location of scene. Uses 'Lazy Loading'.
            scene_id = np.random.randint(low=0, high=len(self.files), size=1).item()

            # - Load scene
            scene = xr.open_dataset(os.path.join(self.options['test_files'], self.files[scene_id]))

            file_id = self.file_ids[self.files[scene_id]]
            # - Extract patches
            try:
                scene_patch = self.random_crop(scene)
            except:
                print(f"Cropping in {self.files[scene_id]} failed.")
                print(f"Scene size: {scene['nersc_sar_primary'].values.shape} for crop shape: ({self.options['patch_size']}, {self.options['patch_size']})")
                print('Skipping scene.')
                continue
            
            if scene_patch is not None:
            
                file_name  = f"test_patch{file_id}_batch{idx}_sample{sample_n}.pt"
                scene_patch_tens = torch.from_numpy(scene_patch).float()
                file_path = os.path.join(self.save_dir, file_name)
                torch.save(scene_patch_tens, file_path)
                print(f"Saved patch to {file_path}")

                # -- Stack the scene patches in patches
                patches[sample_n, :, :, :] = scene_patch
                sample_n += 1 # Update the index.

        # Prepare test data arrays
        x = self.prep_dataset(patches=patches)


        return x
    

class TestDatasetPermute(Dataset):
    """Pytorch dataset for loading full scenes from the ASID ready-to-train challenge dataset for inference."""

    def __init__(self, options, files, test_dir):
        self.options = options
        self.files = files
        self.test_dir = test_dir

    def __len__(self):
        return len(self.files)

    def prep_scene(self, input_scene):
        """
        Upsample low resolution to match charts and SAR resolution. Convert patches from 4D numpy array to 4D torch tensor.
        """
        x = torch.cat((torch.from_numpy(input_scene[self.options['sar_variables']].to_array().values).unsqueeze(0),
                       torch.nn.functional.interpolate(
                           input=torch.from_numpy(input_scene[self.options['amsrenv_variables']].to_array().values).unsqueeze(0),
                           size=input_scene['nersc_sar_primary'].values.shape,
                           mode=self.options['loader_upsampling'])),
                       axis=1)


        return x
    
    def load_labels(self, label_scene):

        y = {chart: torch.tensor(label_scene[chart].values) for chart in self.options['charts']}

        return y
    

    def __getitem__(self, idx):
        """
        Get scene
        """
        input_file = self.files[idx]
        input_path = os.path.join(self.test_dir, input_file)
        input_scene = xr.open_dataset(input_path)
        
        label_file = input_file.replace('.nc', '_reference.nc')
        label_path = os.path.join(self.test_dir, label_file)
        label_scene = xr.open_dataset(label_path)

        x = self.prep_scene(input_scene)
        y = self.load_labels(label_scene)

        name = input_file.split('.')[0]  

        masks = {chart: (y[chart] != self.options['class_fill_values'][chart]).type(torch.bool) for chart in self.options['charts']}

        return x, y, masks, name















