# This script post-process CBEDs into 4D-STEM dataset with different kMax, detector size, intensity fluctuation, detector blur, and noise levels
# Chia-Hao Lee, 2023.12.03 (modified on 2024.03.08 for zenodo update)
# cl2696@cornell.edu

use_np_or_cp = 'cp'

if use_np_or_cp == 'cp':
    import cupy as xp
    from cupyx.scipy import ndimage
else:
    import numpy as xp
    from scipy import ndimage

import os
import h5py
from time import time
from tifffile import imread, imwrite


def getFilename(dir, target_extension, keep_extension):
    '''
    This function go through the folder and return a list of filenames with the specific extension
    '''
    f_list = os.listdir(dir)
    file_list=[]
    for i in f_list:    
        if os.path.splitext(i)[1] == target_extension:
            if keep_extension:
                file_list.append(i)
            else:
                file_list.append(os.path.splitext(i)[0])
    file_list.sort()
    return file_list

def resample_slice(imstack, target_slice_shape, interp_order):
    # Resample the input stack based on the target slice shape
        
    if target_slice_shape != imstack.shape[-2:]:
        imstack_resample = ndimage.zoom(imstack, (1,target_slice_shape[0]/imstack.shape[1], target_slice_shape[1]/imstack.shape[2]), order=interp_order)
    else:
        imstack_resample = imstack.copy()
    return imstack_resample

def normalize_slice(imstack, method):  
    # Normalize the image stack based on the method 'sum' or 'max'
    
    imstack_norm = imstack.copy()  
    if method == 'sum':
        imstack_norm /= imstack_norm.sum(axis=(1,2))[:, None, None]
    elif method == 'max':
        imstack_norm /= imstack_norm.max(axis=(1,2))[:, None, None]
    return imstack_norm

def apply_intensity_variation(imstack, int_var_std, seed):
    # Typical FEG 4D data has intensity std = 0.002, or 0.2 %
    
    if int_var_std != 0:
        xp.random.seed(seed)
        int_variation = xp.random.normal(1, int_var_std, len(imstack))
        imstack_int_var = imstack * int_variation[:,None,None]
    else:
        imstack_int_var = imstack.copy()
    return imstack_int_var

def apply_detector_blur(imstack, sigma):
    # Apply detector blur with a given sigma

    if sigma != 0:
        imstack_blur = ndimage.gaussian_filter(imstack, (0, sigma, sigma))
    else:
        imstack_blur = imstack.copy()
    return imstack_blur

def apply_poisson_noise(imstack, seed):
    # Apply Poisson noise to the image stack
    
    # Although the default_rng generator version is faster than the np.random.poisson()
    # I got some unknown issue with the cupy implementation of default_rng so I'll stick with the original
    
    xp.random.seed(seed)
    imstack_poisson = xp.random.poisson(imstack)
    return imstack_poisson
    
def crop_cbed_square(cbed_stack, crop_shape):
    # Crop the CBED to square
    # The cbed_stack shape is expected to be (...,ky, kx)
    
    cbed_shape = cbed_stack.shape[-2:]
    if cbed_shape != crop_shape:
        crop_length = crop_shape[0]
        center = (cbed_shape[-2]//2, cbed_shape[-1]//2)
        row_start, column_start = center[0] - crop_length//2, center[1] - crop_length//2
        row_end, column_end = center[0] + crop_length//2, center[1] + crop_length//2
        cbed_stack_crop = cbed_stack[..., row_start:row_end, column_start:column_end]
    else:
        cbed_stack_crop = cbed_stack.copy()
    return cbed_stack_crop

# Setup parameters
# Don'try to run all configurations at once due to GPU memory constraint
simu_dir = 'data/test_abTEM_output'
output_dir = 'data/test_ptycho_recon' # Parent output dir, or equvalently the "exp_dir"

potential_modes= ['static']
#['static', 'phonon']
coherence_modes= ['coherent']
#['coherent', 'temporal', 'spatial', 'temporal_spatial']
alpha_dirs= ['alpha_7.5mrad']
#['alpha_10.5mrad', 'alpha_23.1mrad', 'alpha_40.0mrad'] 
cs_dirs= ['Cs_1.0um']
#['Cs_1.0um', 'Cs_1.1mm']

input_simu_shape = (1045, 1047)
crop_shapes = [(1045, 1047)] #(1044//2, 1044//2), (1045//2, 1047//2)
target_shapes = [(128,128)]#, (256,256), (512,512), (1024, 1024)] #,(128,128), (256,256), (512,512), (1024, 1024)
int_var_std = 0.0 #.002 # Typical FEG 4D data has intensity std = 0.002, or 0.2 %
detector_blur_stds = [0] # [0.5, 1, 2] This is essentially the PSF of detector or FT[MTF], usually we do 1 px in EMPAD at 300kV when it's upsampled to 256 x 256
doses = [1e8] #[1e6, 1e8, 1e14] # e-/Ang^2, 

inf_dose_alias = 1e14 # A handy alias for inifinite dose, the if statement would take this dose and bypass the Poisson process
scan_step_size = 1 # Ang

extension = '.tif'
interp_order = 1 # Bilinear will keep the value range, while the default bicubic (order = 3) would produce values outside of the range
save_option = 'both' # ".tif", ".hdf5", "both"
xp_seed = 42


print(f"We're about to generate\
 {len(potential_modes)} x {len(coherence_modes)} x {len(alpha_dirs)} x {len(cs_dirs)} x {len(crop_shapes)} x {len(target_shapes)} x {len(detector_blur_stds)} x {len(doses)} =\
 {len(potential_modes) * len(coherence_modes) * len(alpha_dirs) * len(cs_dirs) * len(crop_shapes) * len(target_shapes) * len(detector_blur_stds) * len(doses)} 4D-STEM datasets\
 from {len(potential_modes) * len(coherence_modes) * len(alpha_dirs) * len(cs_dirs)} simulated datasets ")

# Loop over each dimension to create corresponging hdf5
# Doing the operation at each level is around 2x faster then doing it all in the inner most loop for current loop number configurations
# Took around 1 hr on my workstation with CPU, and 24 min on my P5000
# Most of the time was spent on data i/o so doing it with GPU can only provide moderate speed up unless we do more parallelization with broadcasting and dask to prevent memory issue

for potential_mode in potential_modes:
    for coherence_mode in coherence_modes:
        for alpha_dir in alpha_dirs:
            for cs_dir in cs_dirs:
                # Loading the file takes a while, we want to minimize loading
                mode_dir = f"{potential_mode}_{coherence_mode}"
                input_dir = os.path.join(simu_dir, mode_dir, alpha_dir, cs_dir)
                fname_list = getFilename(input_dir, extension, keep_extension=True)
                fname_list.sort()
                imstack = xp.zeros((len(fname_list), *input_simu_shape), dtype='float32')

                start_load_time = time()
                for i, fname in enumerate(fname_list):
                    imstack[i] = xp.array(imread(os.path.join(input_dir, fname)))
                end_load_time = time()
                print(f"Done loading {input_dir} with {len(fname_list)} files into image stack in {(end_load_time - start_load_time):.2f} seconds!")
                
                for j, crop_shape in enumerate(crop_shapes):
                    imstack_crop = crop_cbed_square(imstack, crop_shape)                        
                    print(f"imstack_crop.shape = {imstack_crop.shape}")
                    
                    for target_shape in target_shapes:
                        imstack_resample = resample_slice(imstack_crop, target_shape, interp_order)
                        print(f"imstack_resample.shape = {imstack_resample.shape}")
                        if interp_order > 1:
                            print(f"Clip {xp.sum(imstack_resample < 0) / imstack_resample.shape[0]} negative px per CBED at 0 after resampling")
                            imstack_resample[imstack_resample<0] = 0 # Clip the negative value, this only happens when interp_order > 1
                        imstack_norm = normalize_slice(imstack_resample, 'sum')
                        imstack_int_var = apply_intensity_variation(imstack_norm, int_var_std, xp_seed)
                            
                        for detector_blur_std in detector_blur_stds:
                            print(f"Adding blur = {detector_blur_std}")
                            imstack_blur = apply_detector_blur(imstack_int_var, detector_blur_std) # Remove the implicit scaling so the value is less confusion

                            for dose in doses:
                                # Due to the px with 0, it seems better to keep simulated CBED in the original unit than the electron count
                                # Otherwise the CBED with large dose will still have some 0 px, which is quite odd
                                # Therefore, I unnormalize the CBED after applying poisson with the original sum value before normalization
                                print(f"Adding dose = {dose}")
                                if dose == inf_dose_alias:
                                    imstack_poisson = imstack_blur.astype('float32') * inf_dose_alias
                                    imstack_poisson = imstack_poisson / inf_dose_alias * imstack_resample.sum(axis = (1,2))[:,None,None]
                                else:
                                    total_electron = dose * scan_step_size**2
                                    imstack_poisson = apply_poisson_noise(imstack_blur * total_electron, xp_seed).astype('float32')
                                    imstack_poisson = imstack_poisson / total_electron * imstack_resample.sum(axis = (1,2))[:,None,None]
                                sub_dir = f'data3D_200kV_df_10nm_{alpha_dir}_{cs_dir}_dp_{target_shape[0]}_blur_{detector_blur_std}px_dose_{dose:.1e}ePerAng2'

                                if use_np_or_cp =='cp':
                                    data3D = imstack_poisson.get()
                                else:
                                    data3D = imstack_poisson

                                if crop_shape[0] == input_simu_shape[0]//2:
                                    output_dir_postfix = output_dir + '_kmax2'
                                elif crop_shape[0] == input_simu_shape[0]:
                                    output_dir_postfix = output_dir + '_kmax4'
                                
                                os.makedirs(os.path.join(output_dir_postfix, mode_dir, sub_dir, '1'), exist_ok=True)
                                fname = f'data_roi_1_Ndp_{target_shape[0]}_dp'
                                file_path = os.path.join(output_dir_postfix, mode_dir, sub_dir,'1', fname)
                                
                                if save_option in {'.tif', 'both'}:
                                    imwrite(file_path + '.tif', data3D)
                                if save_option in {'.hdf5', 'both'}:
                                    with h5py.File(file_path + '.hdf5', 'w') as hf:
                                        hf.create_dataset('/dp', data = data3D)
                                
                                del imstack_poisson, data3D
                                if use_np_or_cp:
                                    xp.get_default_memory_pool().free_all_blocks()
                            del imstack_blur
                            if use_np_or_cp:
                                xp.get_default_memory_pool().free_all_blocks()
                        del imstack_resample, imstack_norm, imstack_int_var
                        if use_np_or_cp:
                            xp.get_default_memory_pool().free_all_blocks()
                    del imstack_crop
                    if use_np_or_cp:
                        xp.get_default_memory_pool().free_all_blocks()
                del imstack
                if use_np_or_cp:
                    xp.get_default_memory_pool().free_all_blocks()

print(f"Done generating {save_option} datasets!")
        
        
        