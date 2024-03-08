# This script generates 4D-STEM dataset with multislice, phonon, spatial, temporal partial coherence, probe position error
# Chia-Hao Lee, 2023.11.29 (modified on 2024.03.08 for zenodo update)
# cl2696@cornell.edu

##########################
##### Import packages ####
##########################

import numpy as np
import ase
import abtem
import os
import dask
import cupy as cp
from datetime import datetime
from tifffile import imwrite
abtem.config.set({"local_diagnostics.progress_bar": False})
abtem.config.set({"device": "gpu"})
abtem.config.set({"dask.chunk-size-gpu" : "2048 MB"})
dask.config.set({"num_workers": 1})


##########################
##### Define functions ###
##########################

def gaussian_kernel(patch_size, center, sigma, normalize = 'sum'):
    """Generate a Gaussian kernel."""
    y, x = np.meshgrid(np.arange(patch_size), np.arange(patch_size))
    kernel = np.exp(-((y - center[0])**2 + (x - center[1])**2) / (2 * sigma ** 2))
    if normalize == 'sum':
        return kernel/np.sum(kernel) # Normalzize summation to 1
    elif normalize == 'max':
        return kernel / np.max(kernel)  # Normalize peak value to 1
    
def set_modes(potential_mode, coherence_mode):
    # Set calculation mode based on user input
    
    # Input:
    # potential_mode: str, 'static', or 'phonon'
    # coherence_mode: str, 'coherent', 'temporal', 'spatial', 'temporal_spatial'
    
    # Output:
    # potential_calc: abTEM potential object
    # probe_calc: abTEM potential object
    # scan_calc: Nx2 or Nx9x2 scan poistion array
    # scan_mode: str    
    
    if potential_mode == 'static':
        potential_calc = potential_static
    elif potential_mode == 'phonon':
        potential_calc = potential_phonon
    else:
        print("Potential mode not implemented!")
        return
    
    if coherence_mode in ['coherent', 'spatial']:
        probe_calc = probe_coherent
    elif coherence_mode in ['temporal', 'temporal_spatial']:
        probe_calc = probe_temporal
    else:
        print("Coherence mode not implemented!")
        return
    
    if coherence_mode in ['spatial', 'temporal_spatial']:
        scan_mode = 'major_sub'
        scan_calc = grid_scan_with_err_subscan_reshape
    else:
        scan_mode = 'major'
        scan_calc = grid_scan_with_err
        
    return potential_calc, probe_calc, scan_calc, scan_mode

def simulate(potential_calc, probe_calc, scan_calc, scan_mode, idx, gaussian_weights_ravel, keep_unweighted_subscan):
    # Run abTEM multislice simulation based on the potential, probe, and scan
    # Also return post-processed cbed_output based on whether the subscans are kept
    # if keep_unweighted_subscan is True, then the output CBED would have 1 extra subscan dimentsion
    
    pixelated_detector = abtem.PixelatedDetector(max_angle='cutoff') # This will cut the output CBED to the kmax_antialias
    cbed_calc = probe_calc.multislice(scan = scan_calc[idx], potential = potential_calc, detectors = pixelated_detector)
    
    if scan_mode == 'major':
        cbed_output = cbed_calc.compute().array
    else:
        if keep_unweighted_subscan:
            cbed_output = cbed_calc.compute().array
        else:
            cbed_output = cp.sum(cp.array(cbed_calc.compute().array) * cp.array(gaussian_weights_ravel[:,None, None]), axis = -3).get()

    # Print information
    if idx == start_index:
        print(f"cbed_calc.shape = {cbed_calc.shape}")
        print(f"cbed_calc_metadata = \n{cbed_calc.axes_metadata}")
        print(f"cbed_output.shape = {cbed_output.shape}")
    
    return cbed_output

def save_cbed(cbed_crop, idx, keep_unweighted_subscan):
    cbed_shape = cbed_crop.shape
    # Save cbed
    for alpha_i in range(cbed_shape[1]):
        alpha_Cs1_dir = f"alpha_{convergence_angles[alpha_i]}mrad/Cs_1.0um"
        alpha_Cs2_dir = f"alpha_{convergence_angles[alpha_i]}mrad/Cs_1.1mm"
        
        # Make sub directories for alpha and Cs
        if idx == start_index:
            os.makedirs(os.path.join(output_dir, mode_dir, alpha_Cs1_dir), exist_ok = True)
            os.makedirs(os.path.join(output_dir, mode_dir, alpha_Cs2_dir), exist_ok = True)
            print(f"Made alpha_Cs directories for {alpha_Cs1_dir}, {alpha_Cs2_dir}")
    
        if keep_unweighted_subscan and len(cbed_shape)> 4:
            for subscan_i in range(cbed_shape[-3]):
                output_filepath_Cs1 = os.path.join(output_dir, mode_dir, alpha_Cs1_dir, f"CBED_{str(idx).zfill(5)}_sub{str(subscan_i).zfill(2)}.tif")
                output_filepath_Cs2 = os.path.join(output_dir, mode_dir, alpha_Cs2_dir, f"CBED_{str(idx).zfill(5)}_sub{str(subscan_i).zfill(2)}.tif")
                imwrite(output_filepath_Cs1, np.float32(cbed_crop[0, alpha_i, subscan_i] / cbed_crop[0, alpha_i, subscan_i].max()))
                imwrite(output_filepath_Cs2, np.float32(cbed_crop[1, alpha_i, subscan_i] / cbed_crop[1, alpha_i, subscan_i].max()))

                        
        else:
            output_filepath_Cs1 = os.path.join(output_dir, mode_dir, alpha_Cs1_dir, f"CBED_{str(idx).zfill(5)}.tif")
            output_filepath_Cs2 = os.path.join(output_dir, mode_dir, alpha_Cs2_dir, f"CBED_{str(idx).zfill(5)}.tif")
            imwrite(output_filepath_Cs1, np.float32(cbed_crop[0, alpha_i] / cbed_crop[0, alpha_i].max()))
            imwrite(output_filepath_Cs2, np.float32(cbed_crop[1, alpha_i] / cbed_crop[1, alpha_i].max()))
    return


##########################
####### Setup params #####
##########################

# Set up random seed
np_seed = 42
np.random.seed(np_seed)

# Atomic model
mx2_formula = 'WSe2'
mx2_phase = '2H'
lattice_constant = 3.297
uc_thickness = 3.376
vacuum_layers = 2
supercell_reps = (38, 22, 1)

# Phonon
num_phonon_configs = 25 # target value: 25
phonon_sigma = 0.1 # Ang

# Potential Sampling
real_space_sampling = 0.08 # unit: Ang, note that kmax_antialias = 1/(3*dx), so if we want to simulate up to kmax = 4.1 1/Ang, we need 1/4.1/3 Ang sampling or slightly finer ~ 0.08 Ang 
dz = 1 # Ang, multislice thickness

# Probe parameters
energy = 200e3 # unit: eV
wavelength = 0.025079 # unit: Ang, this value is only used for display useful information
convergence_angles = [7.50, 9.50]#, 10.5, 12.0, 15.0, 18.0, 20.0, 22.0, 23.1, 24.0, 26.0, 28.0,  30.0, 32.0, 34.0, 36.0, 38.0, 40.0] # unit: mrad, target value: [10.5, 23.1, 40.0]
df = 100 # df, unit: Ang, note the df = -C1,0, so positive defocus is underfocuse just like Kirkland and fold_slice.
C30_list = [1 * 1e-6 * 1e10 , 1.1 * 1e-3 * 1e10] # unit: Ang, note that we convert to m and then Ang. C30 = Cs.
aberrations = {"C30": C30_list}

# Temporal partial coherence
chromatic_aberration = 1 * 1e-3 * 1e10 # unit: Ang, note that we convert to m and then Ang
energy_spread = 0.35 # unit: eV, this is the std so expected FWHM of ZLP would be 2.355*0.35 ~ 0.82 eV
num_df_configs = 9

# Scan configurations
x_scan_num, y_scan_num = 32, 32
scan_step_size = 1 # Unit: Ang
pos_error_std = 0.10 # Unit: Ang

# Spatial parital coherence
blur_kernel_size = 3 # This would require a 3x3 grid and it's essentially 9 scan positions
sub_scan_probe_shift = 0.8/2.355 # Unit: Ang, the idea is to get equivalently 0.8 Ang FWHM source size blurring. FWHM = 2.355 sigma.
spatial_blur_std = 1 # Unit: sub scan step size


# Control the simulation behavior
potential_modes= ['static'] #, 'phonon']
coherence_modes= ['coherent'] #, 'temporal', 'spatial', 'temporal_spatial']
keep_unweighted_subscan = False

# Artifical parallelization across different cluster node (use scripts in `template2scripts`)
start_index= 0
end_index= 1024 # Not included

# Output directory
output_dir = 'data/test_abTEM_output'

##########################
##### Initiate abTEM #####
##########################

# Prepare potentials
atoms = ase.build.mx2(formula=mx2_formula, kind=mx2_phase, a=lattice_constant, thickness=uc_thickness, vacuum=vacuum_layers) # a: lattice constant, thickness: chalcogen intralayer distance, vacuum = vacuum layer thickness. All unit in Ang.
atoms_sc = abtem.orthogonalize_cell(atoms) * supercell_reps # lx:ly = 1:sqrt(3)
print(f'atoms_sc.cell = {atoms_sc.cell} Ang') # Unit: Ang

# Include frozen phonon configuration
phonon_seed = np.random.randint(0,1000, num_phonon_configs)
frozen_phonons = abtem.FrozenPhonons(atoms_sc, num_configs=num_phonon_configs, sigmas=phonon_sigma, seed = phonon_seed) # sigmas is in unit of Ang.
print(f'phonon_seed = {phonon_seed}')

potential_static = abtem.Potential(atoms_sc, sampling = real_space_sampling, slice_thickness=dz)
potential_phonon = abtem.Potential(frozen_phonons, sampling = real_space_sampling, slice_thickness=dz)
print(f'potential_static.shape = {potential_static.shape}')
print(f'potential_phonon.shape = {potential_phonon.shape}')

# Prepare probes
kmax_antialias = 1/real_space_sampling/3 # 1/Ang #The kmax_antialiasing = 4.166 Ang-1
alpha_max_antialias = wavelength * kmax_antialias # rad
focal_spread = chromatic_aberration * energy_spread / energy

defocus_distribution = abtem.distributions.gaussian(
    center = df,
    standard_deviation=focal_spread,
    num_samples=num_df_configs,
    sampling_limit=2,
    ensemble_mean=True)
print(f"Energy = {energy/1e3} kV, rel. wavelength = {wavelength} Ang")
print(f"CBED collection kmax = {kmax_antialias} 1/Ang, collection alpha_max = {alpha_max_antialias*1000} mrad")
print(f"Focal spread = {focal_spread} Å")
print(f"defocus distribution = {np.array(defocus_distribution)}")

probe_coherent = abtem.Probe(energy=energy, semiangle_cutoff=convergence_angles, defocus=df,                   **aberrations)
probe_temporal = abtem.Probe(energy=energy, semiangle_cutoff=convergence_angles, defocus=defocus_distribution, **aberrations)
probe_coherent.grid.match(potential_phonon)
probe_temporal.grid.match(potential_phonon)

# Prepare scan positions
grid_scan = scan_step_size * np.array([(x, y) for y in range(0, y_scan_num) for x in range(0, x_scan_num) ])
pos_error = np.random.normal(scale=pos_error_std, size = grid_scan.shape)
grid_scan_with_err = grid_scan + pos_error

print(f"First 5 grid_scan position (x, y) or (hori, vertic) = \n{grid_scan[:5]}")
print(f"Scan number x, y = {x_scan_num, y_scan_num}")
print(f"Scan step size = {scan_step_size} Ang or {scan_step_size / real_space_sampling:.02f} px")
print(f"Scan area = {scan_step_size * x_scan_num} x {scan_step_size * y_scan_num} Ang^2 or {scan_step_size / real_space_sampling * x_scan_num:.02f} x {scan_step_size / real_space_sampling * y_scan_num:.02f} px^2")
print(f"Position error of std = {pos_error_std} Ang of {pos_error_std / real_space_sampling:.02f} px is applied")

gaussian_weights = gaussian_kernel(blur_kernel_size, (blur_kernel_size//2, blur_kernel_size//2), spatial_blur_std, normalize='sum')
gaussian_weights_ravel = gaussian_weights.ravel()
print(f"gaussian_weights = \n{gaussian_weights.round(3)}")
print(f"Sub scan probe shift = {sub_scan_probe_shift:.2f} Ang or {sub_scan_probe_shift / real_space_sampling:.2f} px")


grid_scan_with_err_subscan = np.array([
    [grid_scan_with_err[n,0] + i * sub_scan_probe_shift, grid_scan_with_err[n,1] + j * sub_scan_probe_shift] 
    for n in range(len(grid_scan_with_err))
    for j in range(-blur_kernel_size // 2 + 1, (blur_kernel_size // 2) + 1)
    for i in range(-blur_kernel_size // 2 + 1, (blur_kernel_size // 2) + 1)])
grid_scan_with_err_subscan_reshape = np.reshape(grid_scan_with_err_subscan, ((len(grid_scan_with_err), len(gaussian_weights_ravel), -1)))


##########################
##### Main execution #####
##########################
for potential_mode in potential_modes:
    for coherence_mode in coherence_modes:
        
        # Set modes and make directory
        potential_calc, probe_calc, scan_calc, scan_mode = set_modes(potential_mode, coherence_mode)
        mode_dir = f"{potential_mode}_{coherence_mode}"
        os.makedirs(os.path.join(output_dir, mode_dir), exist_ok=True)
        print(f"Made mode directory for {mode_dir}")
        start_time = datetime.now()
        print(f'\nStart simulating potential mode = {potential_mode}, coherence mode = {coherence_mode}, scan_mode = {scan_mode} at {start_time.strftime("%H:%M:%S")}')

        # Start simulating for each major scan position
        for idx in range(start_index, end_index):
            print(f"Simulating position {idx}")
            print(f"Position(s) = {scan_calc[idx]} Ang")
            
            # Start generating CBED at each idx
            cbed_output = simulate(potential_calc, probe_calc, scan_calc, scan_mode, idx, gaussian_weights_ravel, keep_unweighted_subscan)
            save_cbed(cbed_output, idx, keep_unweighted_subscan)
            
        end_time = datetime.now()
        print(f'Finish simulating potential mode = {potential_mode}, coherence mode = {coherence_mode}, scan_mode = {scan_mode} at {end_time.strftime("%H:%M:%S")}')