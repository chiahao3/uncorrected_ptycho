% This script 1) prepares experimental electron ptycho. data for PtychoShelves
% and 2) run the electron ptychographic reconstruction. It's a combination of
% sample scripts from Yi Jiang's github repo: fold_slice 
% (https://github.com/yijiang1/fold_slice)
% Last modified by Chia-Hao Lee@Cornell, 2024/03/07

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%% Part I: Prepare the data and the initial probe %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 0: Add path, clear variables
% Set paths so that the packpages can be found without changing working
% directories (i.e. run scripts elsewhere)

cSAXS_matlab_path = './fold_slice_240207';
ptycho_matlab_path = './fold_slice_240207/ptycho';
addpath(cSAXS_matlab_path);
addpath(ptycho_matlab_path);
addpath(strcat(ptycho_matlab_path,'/utils'));
addpath(strcat(ptycho_matlab_path,'/utils_electron'));
gpuDevice

%% Step 1: Setup parameters
% Data import and probe generation
exp_dir = 'data/'; %change this for different experiment
scan_dir = 'Fig_2h_10.3mrad_Talos'; % change this for different scans
base_path = strcat(exp_dir, scan_dir, '/'); 
jobID = strcat('WSe2_', scan_dir); %This will be used for the filename of param.m
dim_x = 128; %EMPAD pixel number
dim_y = 130; %Extra rows of EMPAD, will crop later
N_scan_x = 128; %Real space scan positions
N_scan_y = 128;
dScanX = 1; %scan downsample factor, 2 is to load every other scan position
dScanY = 1; %scan downsample factor, 2 is to load every other scan position 
Np_crop_pad = [128,128]; % size of diffraction patterns / probe used during reconstruction. can crop/pad to 64/256
resample_factor = 2; %DP upsample factor
scan_number = 1; %Ptychoshelves needs
final_scan = [N_scan_x/dScanX, N_scan_y/dScanY]; %Final number of scan positions
final_dp_size = round(resample_factor * Np_crop_pad); %Final dimension of diffraction pattern

% Microscope and acquisition parameters
df = 0*10*-1*0.798; %defocus in angstrom, the sign is oppisite to UI defocus, and there's a 1.43 factor for actual defocus at 80kV for Themis
ADU = 393; % Depends on kV, calibrated from the orignal EMPAD paper: doi:10.1017/S1431927615015664, 80kV: 151, 200kV: 393
voltage = 200;
alpha0 = 10.34; %convergence semi angle # 18->18.44, 25.2->24.92, 36->36.4, 54->55.45
rbf = 6.87; %radius of central disk in px at given camera length. 16 - cl 230 mm; 
rot_ang = 0; %angle between cbed and scan coord.
scan_step_size = 0.5327; % For 128x128 scan positions, 1.182 - 5.1 Mx; 0.839 - 7.2 Mx; 0.605 - 10 Mx; 0.429 - 14.5Mx; 0.297 - 20.5 Mx; 0.208 - 29 Mx; 0.147 - 41 Mx %angstrom

% Reconstruction parameters
data_descriptor = 'WSe2_'; %A short string that describe data when sending notifications
Niter = 1000;
Niter_save_results = 100;
Niter_plot_results = 100;
Nprobe = 10; % # of probe modes
variable_probe_modes = 2; % 2 if possible # of modes for variable probe correction
grouping = 128; % group size. small -> better convergence but longer time/iteration
N_pos_corr = 0; % iteration number to start position correction. inf means no position correction
GPU_solver = 'MLs'; % choose GPU solver: DM, ePIE, hPIE, MLc, Mls, -- recommended are MLc and MLs
errmetric = 'L1';            % optimization likelihood - poisson, L1
recon_time = datestr(now, 'yyyy-mm-dd_HH-MM-SS');

initial_probe_file = strcat(base_path, sprintf('%i/init_probe_%i.mat', scan_number, final_dp_size(1)));
%init_recon_file = '/scratch/users/chiahao3/20230119_TEM_Talos_200kV_WSe2_16_EMPAD/0036_step128_20.5Mx_cl98mm_10.5mrad_df0_19pA/3/roi3_Ndp256_step128/MLs_L1_p15_g256_pc500_noModel_updW100_vp2_vi_mm/Niter9000.mat';
use_previous_probe = false; %set false when you use a new data set
use_previous_object = false; %false for new data
use_previous_position = false; % false for new data
probe_change_start = 1;      % Start updating probe at this iteration number
object_change_start = 1;       % Start updating object at this iteration number

% Flags
if Np_crop_pad(1)~=128
    if_crop_pad = true;
else
    if_crop_pad = false;   
end

if resample_factor~=1
    if_resample = true;
    rbf = rbf * resample_factor; 
else
    if_resample = false;
end

save(strcat(base_path, jobID, sprintf('_params_%s.mat', recon_time))); %Save the parameters
disp(sprintf('Started at %s', recon_time))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%% Shouldn't need to change anything below for common usage %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Step 2: load data
scan_name = sprintf('scan_x%i_y%i.raw', N_scan_x, N_scan_y);
fin = fread(fopen(strcat(base_path, scan_name),'r'), dim_x * dim_y * N_scan_x * N_scan_y,'float32=>float32'); % Load as 1-D vector with float32 and save as float32
fin_reshape = reshape(fin, dim_x, dim_y, N_scan_x, N_scan_y); %Matlab fill its column first, so the DP and scan pattern would be automatically transposed
cbed = double(fin_reshape(1:128,1:128,1:dScanX:end,1:dScanY:end)); % crop the CBED into 128x128

%% Step 3: Pre-process the size of the diffraction patterns (pad, crop, resample)

[ndpx, ndpy, npx, npy] = size(cbed); % get the raw data dimentsion

% Step 3-1: Crop / pad the cbed based on Np_crop_pad

if ndpy < Np_crop_pad(1) % pad zeros
    dp=padarray(cbed,[(Np_crop_pad(1)-ndpy)/2,(Np_crop_pad(2)-ndpx)/2,0,0],0,'both');
else
    dp=crop_pad(cbed, Np_crop_pad);
end

% Step 3-2: Resample the cbed based on resample_factor

if resample_factor ~= 1
    dp = reshape(dp, Np_crop_pad(1), Np_crop_pad(2), []);%reshape to 3D so that we can use the built-in imresize3 instead of imresizen
    dp = imresize3(dp, [final_dp_size(1), final_dp_size(2), npx*npy]);
else
	dp = reshape(dp, Np_crop_pad(1), Np_crop_pad(2), []);
end

% Normalizing the cbed intensity
dp = dp / ADU; % convert to electron count
Itot=mean(squeeze(sum(sum(dp,1),2))); %need this for normalizting initial probe
dp=reshape(dp, final_dp_size(1), final_dp_size(2), npx, npy);
[Ndpx, Ndpy, Npx, Npy] = size(dp); %get the final data dimension

% calculate pixel size (1/A) in diffraction plane
[~,lambda]=electronwavelength(voltage);
dk=alpha0/1e3/rbf/lambda; %%% PtychoShelves script needs this %%%
dp=reshape(dp, Ndpx, Ndpy, []); %PtychoShelves actually takes this 3D shape

%% Step 4: save 4D-STEM in a .hdf5 file (needed by Ptychoshelves)

save_dir = strcat(base_path,num2str(scan_number),'/');
mkdir(save_dir)
roi_label = sprintf('%i_Ndp%i_step%i', scan_number, Ndpx, Npx);
saveName = strcat('data_roi',roi_label,'_dp.hdf5');
dp_dir = strcat(save_dir,saveName);

if isfile(dp_dir)
    delete(dp_dir);
end    
dp = single(dp);
h5create(dp_dir, '/dp', size(dp),'ChunkSize',[size(dp,1), size(dp,2), 1],'Deflate',4)
h5write(dp_dir, '/dp', dp)
disp(strcat(saveName, ' has been saved!'))

%% Step 5: prepare an save initial probe
dx=1/Ndpx/dk; %% pixel size in real space (angstrom)

par_probe = {};
par_probe.df = df;
par_probe.voltage = voltage;
par_probe.alpha_max = alpha0;
par_probe.plotting = true;
probe = make_tem_probe(dx, Ndpx, par_probe);

probe=probe/sqrt(sum(sum(abs(probe.^2))))*sqrt(Itot)/sqrt(Ndpx*Ndpy);
probe=single(probe);

% add parameters for PtychoShelves
p = {};
p.binning = false;
p.detector.binning = false;

% save initial probe
save(strcat(save_dir,sprintf('/init_probe_%i.mat', Ndpx)),'probe','p')

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% Part II: Prepare and run the reconstruction %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Step 6: initialize data parameters
p = struct();
p.   verbose_level = 3;                            % verbosity for standard output (0-1 for loops, 2-3 for testing and adjustments, >= 4 for debugging)
p.   use_display = Niter_plot_results< Niter;                                      % global switch for display, if [] then true for verbose > 1
p.   scan_number = scan_number;                                    % Multiple scan numbers for shared scans

% Geometry
p.   z = 1;                                             % Distance from object to detector. Always 1 for electron ptycho
p.   asize = [Ndpx,Ndpx];                                     % Diffr. patt. array size
p.   ctr = [fix(Ndpx/2)+1, fix(Ndpx/2)+1];                                       % Diffr. patt. center coordinates (y,x) (empty means middle of the array); e.g. [100 207;100+20 207+10];
p.   beam_source = 'electron';                         % Added by YJ for electron pty. Use relativistic corrected formula for wavelength. Also change the units on figures
%p.   dk = dk;                                          % Added by YJ. dk is the pixel size in cbed (1/A). This is used to determine pixel size in electron ptycho
p.   d_alpha = alpha0/rbf;                              % Added by YJ. d_alpha is the pixel size in cbed (mrad). This is used to determine pixel size in electron ptycho
p.   prop_regime = 'farfield';                              % propagation regime: nearfield, farfield (default), !! nearfield is supported only by GPU engines 
p.   focus_to_sample_distance = [];                         % sample to focus distance, parameter to be set for nearfield ptychography, otherwise it is ignored 
p.   energy = voltage;                                           % Energy (in keV), leave empty to use spec entry mokev

%p.   affine_angle = 0;                                     % Not used by ptycho_recons at all. This allows you to define a variable for the affine matrix below and keep it in p for future record. This is used later by the affine_matrix_search.m script
%p.   affine_matrix = [1 , 0; 0, 1] ; % Applies affine transformation (e.g. rotation, stretching) to the positions (ignore by = []). Convention [yn;xn] = M*[y;x].
affine_mat  = compose_affine_matrix(1, 0, rot_ang, 0);
p.   affine_matrix = affine_mat ; % Applies affine transformation (e.g. rotation, stretching) to the positions (ignore by = []). Convention [yn;xn] = M*[y;x].

% Scan meta data
p.   src_metadata = 'none';                                 % source of the meta data, following options are supported: 'spec', 'none' , 'artificial' - or add new to +scan/+meta/
p.   queue.lockfile = false;                                % If true writes a lock file, if lock file exists skips recontruction

% Data preparation
p.   detector.name = 'empad';                           % see +detectors/ folder 
p.   detector.check_2_detpos = [];                          % = []; (ignores)   = 270; compares to dettrx to see if p.ctr should be reversed (for OMNY shared scans 1221122), make equal to the middle point of dettrx between the 2 detector positions
p.   detector.data_prefix = '';                             % Default using current eaccount e.g. e14169_1_
p.   detector.binning = false;                              % = true to perform 2x2 binning of detector pixels, for binning = N do 2^Nx2^N binning
p.   detector.upsampling = false;                           % upsample the measured data by 2^data_upsampling, (transposed operator to the binning), it can be used for superresolution in nearfield ptychography or to account for undersampling in a far-field dataset
p.   detector.burst_frames = 1;                             % number of frames collected per scan position

p.   prepare.data_preparator = 'matlab_aps';                % data preparator; 'python' or 'matlab' or 'matlab_aps'
p.   prepare.auto_prepare_data = true;                      % if true: prepare dataset from raw measurements if the prepared data does not exist
p.   prepare.force_preparation_data = true;                 % Prepare dataset even if it exists, it will overwrite the file % Default: @prepare_data_2d
p.   prepare.store_prepared_data = false;                    % store the loaded data to h5 even for non-external engines (i.e. other than c_solver)
p.   prepare.prepare_data_function = '';                    % (used only if data should be prepared) custom data preparation function handle;
p.   prepare.auto_center_data = false;                      % if matlab data preparator is used, try to automatically center the diffraction pattern to keep center of mass in center of diffraction

p.   src_positions = 'matlab_pos';                           % 'spec', 'orchestra', 'load_from_file', 'matlab_pos' (scan params are defined below) or add new position loaders to +scan/+positions/
p.   positions_file = [''];    %Filename pattern for position files, Example: ['../../specES1/scan_positions/scan_%05d.dat']; (the scan number will be automatically filled in)

if use_previous_position
    p.   scan.type = 'custom_GPU';     
    p.   scan.custom_positions_source = init_recon_file; % custom: a string name of a function that defines the positions; also accepts mat file with entry 'pos', see +scans/+positions/+mat_pos.m
else
    % scan parameters for option src_positions = 'matlab_pos';
    p.   scan.type = 'raster';                                  % {'round', 'raster', 'round_roi', 'custom'}
    p.   scan.custom_positions_source = '';
end
p.   scan.roi_label = roi_label;                            % For APS data
p.   scan.format = '%01d';                      % For APS data format for scan directory generation
p.   scan.radius_in = 0;                                    % round scan: interior radius of the round scan
p.   scan.radius_out = 5e-6;                                % round scan: exterior radius of the round scan
p.   scan.nr = 10;                                          % round scan: number of intervals (# of shells - 1)
p.   scan.nth = 3;                                          % round scan: number of points in the first shell
p.   scan.lx = 20e-6;                                       % round_roi scan: width of the roi
p.   scan.ly = 20e-6;                                       % round_roi scan: height of the roi
p.   scan.dr = 1.5e-6;                                      % round_roi scan: shell step size
p.   scan.nx = Npx;      %size(dp,3)                                  % raster scan: number of steps in x
p.   scan.ny = Npy;                                                   % raster scan: number of steps in y
p.   scan.step_size_x = dScanX*scan_step_size;                               % raster scan: step size (grid spacing)
p.   scan.step_size_y = dScanY*scan_step_size;                               % raster scan: step size (grid spacing)
p.   scan.custom_flip = [1,1,0];                            % raster scan: apply custom flip [fliplr, flipud, transpose] to positions- similar to eng.custom_data_flip in GPU engines. Added by ZC.
p.   scan.step_randn_offset = 0;                            % raster scan: relative random offset from the ideal periodic grid to avoid the raster grid pathology 
p.   scan.b = 0;                                            % fermat: angular offset
p.   scan.n_max = 1e4;                                      % fermat: maximal number of points generated 
p.   scan.step = 0.5e-6;                                      % fermat: step size 
p.   scan.cenxy = [0,0];                                    % fermat: position of center offset 
p.   scan.roi = [];                                         % Region of interest in the object [xmin xmax ymin ymax] in meters. Points outside this region are not used for reconstruction.
                                                            %  (relative to upper corner for raster scans and to center for round scans)    
                                                            % custom: a string name of a function that defines the positions; also accepts mat file with entry 'pos', see +scans/+positions/+mat_pos.m
p.   scan.custom_params = [];                               % custom: the parameters to feed to the custom position function.

% I/O
p.   prefix = '';                                              % For automatic output filenames. If empty: scan number
p.   suffix = strcat('ML_recon');              % Optional suffix for reconstruction 
p.   scan_string_format = '%01d';                  % format for scan string generation, it is used e.g for plotting and data saving 

%%%p.   base_path = '../../';                                  % base path : used for automatic generation of other paths 
p.   base_path = base_path;     % base path : used for automatic generation of other paths 
p.   specfile = '';                                         % Name of spec file to get motor positions and check end of scan, defaut is p.spec_file == p.base_path;
p.   ptycho_matlab_path = ptycho_matlab_path;               % cSAXS ptycho package path
p.   cSAXS_matlab_path = cSAXS_matlab_path;                 % cSAXS base package path
p.   raw_data_path{1} = '';                                 % Default using compile_x12sa_filename, used only if data should be prepared automatically
p.   prepare_data_path = '';                                % Default: base_path + 'analysis'. Other example: '/afs/psi.ch/project/CDI/cSAXS_project/analysis2/'; also supports %u to insert the scan number at a later point (e.g. '/afs/psi.ch/project/CDI/cSAXS_project/analysis2/S%.5u')
p.   prepare_data_filename = [];                            % Leave empty for default file name generation, otherwise use [sprintf('S%05d_data_%03dx%03d',p.scan_number(1), p.asize(1), p.asize(2)) p.prep_data_suffix '.h5'] as default 
p.   save_path{1} = '';                                     % Default: base_path + 'analysis'. Other example: '/afs/psi.ch/project/CDI/cSAXS_project/analysis2/'; also supports %u to insert the scan number at a later point (e.g. '/afs/psi.ch/project/CDI/cSAXS_project/analysis2/S%.5u')
p.   io.default_mask_file = '';                             % load detector mask defined in this file instead of the mask in the detector packages, (used only if data should be prepared) 
p.   io.default_mask_type = 'binary';                       % (used only if data should be prepared) ['binary', 'indices']. Default: 'binary' 
p.   io.file_compression = 0;                               % reconstruction file compression for HDF5 files; 0 for no compression
p.   io.data_compression = 3;                               % prepared data file compression for HDF5 files; 0 for no compression
p.   io.load_prep_pos = false;                              % load positions from prepared data file and ignore positions provided by metadata

p.   io.data_descriptor = data_descriptor;                     %added by YJ. A short string that describe data when sending notifications 
p.   io.phone_number = '';                      % phone number for sending messages
p.   io.send_failed_scans_SMS = false;                       % send message if p.queue_max_attempts is exceeded
p.   io.send_finished_recon_SMS = false;                    % send message after the reconstruction is completed
p.   io.send_crashed_recon_SMS = false;                     % send message if the reconstruction crashes
p.   io.SMS_sleep = 1800;                                   % max 1 message per SMS_sleep seconds
p.   io.script_name = mfilename;                             % added by YJ. store matlab script name

p.   artificial_data_file = 'template_artificial_data';     % artificial data parameters, set p.src_metadata = 'artificial' to use this template

%% Step 7: Initial reconstruction parameters
% Initial iterate object
if use_previous_object
    p.   model_object = false;                                   % Use model object, if false load it from file 
    p.   model.object_type = 'rand';                            % specify how the object shall be created; use 'rand' for a random initial guess; use 'amplitude' for an initial guess based on the prepared data
    p.   initial_iterate_object_file{1} = init_recon_file;                   %  use this mat-file as initial guess of object, it is possible to use wild characters and pattern filling, example: '../analysis/S%05i/wrap_*_1024x1024_1_recons*'
else
    p.   model_object = true;                                   % Use model object, if false load it from file 
    p.   model.object_type = 'rand';                            % specify how the object shall be created; use 'rand' for a random initial guess; use 'amplitude' for an initial guess based on the prepared data
    p.   initial_iterate_object_file{1} = '';                   %  use this mat-file as initial guess of object, it is possible to use wild characters and pattern filling, example: '../analysis/S%05i/wrap_*_1024x1024_1_recons*'
end

% Initial iterate probe
p.   model_probe = false;                                   % Use model probe, if false load it from file 
p.   model.probe_alpha_max = alpha0;                          % Modal STEM probe's aperture size
p.   model.probe_df = 0;                                 % Modal STEM probe's defocus
p.   model.probe_c3 = 0;                                    % Modal STEM probe's third-order spherical aberration in angstrom (optional)
p.   model.probe_c5 = 0;                                    % Modal STEM probe's fifth-order spherical aberration in angstrom (optional)
p.   model.probe_c7 = 0;                                    % Modal STEM probe's seventh-order spherical aberration in angstrom (optional)
p.   model.probe_f_a2 = 0;                                  % Modal STEM probe's twofold astigmatism in angstrom (optional)
p.   model.probe_theta_a2 = 0;                              % Modal STEM probe's twofold azimuthal orientation in radian (optional)
p.   model.probe_f_a3 = 0;                                  % Modal STEM probe's threefold astigmatism in angstrom (optional)
p.   model.probe_theta_a3 = 0;                              % Modal STEM probe's threefold azimuthal orientation in radian (optional)
p.   model.probe_f_c3 = 0;                                  % Modal STEM probe's coma in angstrom (optional)
p.   model.probe_theta_c3 = 0;                              % Modal STEM probe's coma azimuthal orientation in radian (optional)

if use_previous_probe
    p.   model_probe = false;
    p.   initial_probe_file = init_recon_file;
    p.   normalize_init_probe = false;                           % Added by YJ. Can be used to disable normalization of initial probes
    p.   ortho_probes = true;                                   % orthogonalize probes after each engine
else
    p.   initial_probe_file = initial_probe_file; 
	p.   normalize_init_probe = true;                           % Added by YJ. Can be used to disable normalization of initial probes
    p.   ortho_probes = true;                                   % orthogonalize probes after each engine
end
p.   probe_file_propagation = 0.0e-3;                            % Distance for propagating the probe from file in meters, = 0 to ignore

% Shared scans - Currently working only for sharing probe and object
p.   share_probe  = 0;                                      % Share probe between scans. Can be either a number/boolean or a list of numbers, specifying the probe index; e.g. [1 2 2] to share the probes between the second and third scan. 
p.   share_object = 0;                                      % Share object between scans. Can be either a number/boolean or a list of numbers, specifying the object index; e.g. [1 2 2] to share the objects between the second and third scan. 

% Modes
p.   probe_modes  = Nprobe;                                 % Number of coherent modes for probe
p.   object_modes = 1;                                      % Number of coherent modes for object

% Mode starting guess
p.   mode_start_pow = 0.02;                               % Normalized intensity on probe modes > 1. Can be a number (all higher modes equal) or a vector
p.   mode_start = 'herm';                                   % (for probe) = 'rand', = 'herm' (Hermitian-like base), = 'hermver' (vertical modes only), = 'hermhor' (horizontal modes only)
p.   ortho_probes = true;                                   % orthogonalize probes after each engine

%% Step 8: initialize Plot, save and analyze parameters
p.   plot.prepared_data = false;                         % plot prepared data
p.   plot.interval = [];                                    % plot each interval-th iteration, does not work for c_solver code
p.   plot.log_scale = [0 0];                                % Plot on log scale for x and y
p.   plot.realaxes = true;                                  % Plots show scale in microns
p.   plot.remove_phase_ramp = false;                        % Remove phase ramp from the plotted / saved phase figures 
p.   plot.fov_box = false;                                   % Plot the scanning FOV box on the object (both phase and amplitude)
p.   plot.fov_box_color = 'r';                              % Color of the scanning FOV box
p.   plot.positions = true;                                 % Plot the scanning positions
p.   plot.mask_bool = true;                                 % Mask the noisy contour of the reconstructed object in plots
p.   plot.windowautopos = true;                             % First plotting will auto position windows
p.   plot.obj_apod = false;                                 % Apply apodization to the reconstructed object;
p.   plot.prop_obj = 0;                                     % Distance to propagate reconstructed object before plotting [m]
p.   plot.show_layers = true;                               % show each layer in multilayer reconstruction 
p.   plot.show_layers_stack = false;                        % show each layer in multilayer reconstruction by imagesc3D
p.   plot.object_spectrum = [];                             % Plot propagated object (FFT for conventional ptycho); if empty then default is false if verbose_level < 3 and true otherwise
p.   plot.probe_spectrum = [];                              % Plot propagated probe (FFT for conventional ptycho); if empty then default is false if verbose_level < 3 and true otherwise
p.   plot.conjugate = false;                                % plot complex conjugate of the reconstruction 
p.   plot.horz_fact = 2.5;                                  % Scales the space that the ptycho figures take horizontally
p.   plot.FP_maskdim = 180e-6;                              % Filter the backpropagation (Fourier Ptychography)
p.   plot.calc_FSC = false;                                 % Calculate the Fourier Shell correlation for 2 scans or compare with model in case of artificial data tests 
p.   plot.show_FSC = false;                                 % Show the FSC plots, including the cropped FOV
p.   plot.residua = false;                                  % highlight phase-residua in the image of the reconstructed phase

p.   save.external = true;                             % Use a new Matlab session to run save final figures (saves ~6s per reconstruction). Please be aware that this might lead to an accumulation of Matlab sessions if your single reconstruction is very fast.
p.   save.store_images = false;                              % Write preview images containing the final reconstructions in [p.base_path,'analysis/online/ptycho/'] if p.use_display = 0 then the figures are opened invisible in order to create the nice layout. It writes images in analysis/online/ptycho
p.   save.store_images_intermediate = false;                % save images to disk after each engine
p.   save.store_images_ids = 1:4;                           % identifiers  of the figure to be stored, 1=obj. amplitude, 2=obj. phase, 3=probes, 4=errors, 5=probes spectrum, 6=object spectrum
p.   save.store_images_format = 'png';                      % data type of the stored images jpg or png 
p.   save.store_images_dpi = 150;                           % DPI of the stored bitmap images 
p.   save.exclude = {'fmag', 'fmask', 'illum_sum'};         % exclude variables to reduce the file size on disk
p.   save.save_reconstructions_intermediate = false;        % save final object and probes after each engine
p.   save.save_reconstructions = false;                      % save reconstructed object and probe when full reconstruction is finished 
p.   save.output_file = 'h5';                               % data type of reconstruction file; 'h5' or 'mat'

%% Step 9: initialize reconstruction parameters %%%%%%%%%%%%%%%%%%%%
% --------- GPU engines  -------------   See for more details: Odstr?il M, et al., Optics express. 2018 Feb 5;26(3):3108-23.
eng = struct();                        % reset settings for this engine
eng. name = 'GPU';    
eng. use_gpu = true;                   % if false, run CPU code, but it will get very slow 
eng. keep_on_gpu = true;               % keep data + projections on GPU, false is useful for large data if DM is used
eng. compress_data = false;             % use automatic online memory compression to limit need of GPU memory
eng. gpu_id = [];                      % default GPU id, [] means choosen by matlab
eng. check_gpu_load = true;            % check available GPU memory before starting GPU engines 

% general
eng. number_iterations = Niter;          % number of iterations for selected method 
eng. asize_presolve = [];      % crop data to 'asize_presolve' size to get low resolution estimate that can be used in the next engine as a good initial guess 
eng. align_shared_objects = false;     % before merging multiple unshared objects into one shared, the object will be aligned and the probes shifted by the same distance -> use for alignement and shared reconstruction of drifting scans  

eng. method = GPU_solver;                   % choose GPU solver: DM, ePIE, hPIE, MLc, Mls, -- recommended are MLc and MLs
eng. opt_errmetric = errmetric;            % optimization likelihood - poisson, L1
eng. grouping = grouping;                    % size of processed blocks, larger blocks need more memory but they use GPU more effeciently, !!! grouping == inf means use as large as possible to fit into memory 
                                       % * for hPIE, ePIE, MLs methods smaller blocks lead to faster convergence, 
                                       % * for MLc the convergence is similar 
                                       % * for DM is has no effect on convergence
eng. probe_modes  = p.probe_modes;                % Number of coherent modes for probe
eng. object_change_start = object_change_start;          % Start updating object at this iteration number
eng. probe_change_start = probe_change_start;           % Start updating probe at this iteration number

% regularizations
eng. reg_mu = 0;                       % Regularization (smooting) constant ( reg_mu = 0 for no regularization)
eng. delta = 0;                        % press values to zero out of the illumination area in th object, usually 1e-2 is enough 
eng. positivity_constraint_object = 0; % enforce weak (relaxed) positivity in object, ie O = O*(1-a)+a*|O|, usually a=1e-2 is already enough. Useful in conbination with OPRP or probe_fourier_shift_search  

eng. apply_multimodal_update = true; % apply all incoherent modes to object, it can cause isses if the modes collect some crap 
eng. probe_backpropagate = 0;         % backpropagation distance the probe mask, 0 == apply in the object plane. Useful for pinhole imaging where the support can be applied  at the pinhole plane
eng. probe_support_radius = [];       % Normalized radius of circular support, = 1 for radius touching the window    
eng. probe_support_fft = false;       % assume that there is not illumination intensity out of the central FZP cone and enforce this contraint. Useful for imaging with focusing optics. Helps to remove issues from the gaps between detector modules.

% basic recontruction parameters 
% PIE / ML methods                    % See for more details: Odstr?il M, et al., Optics express. 2018 Feb 5;26(3):3108-23.
eng. beta_object = 1;                 % object step size, larger == faster convergence, smaller == more robust, should not exceed 1
eng. beta_probe = 1;                  % probe step size, larger == faster convergence, smaller == more robust, should not exceed 1
eng. delta_p = 0.1;                   % LSQ dumping constant, 0 == no preconditioner, 0.1 is usually safe, Preconditioner accelerates convergence and ML methods become approximations of the second order solvers 
eng. momentum = 0;                    % add momentum acceleration term to the MLc method, useful if the probe guess is very poor or for acceleration of multilayer solver, but it is quite computationally expensive to be used in conventional ptycho without any refinement. 
                                      % The momentum method works usually well even with the accelerated_gradients option.  eng.momentum = multiplication gain for velocity, eng.momentum == 0 -> no acceleration, eng.momentum == 0.5 is a good value
                                      % momentum is enabled only when par.Niter < par.accelerated_gradients_start;
eng. accelerated_gradients_start = inf; % iteration number from which the Nesterov gradient acceleration should be applied, this option is supported only for MLc method. It is very computationally cheap way of convergence acceleration. 

% DM
eng. pfft_relaxation = 0.05;          % Relaxation in the Fourier domain projection, = 0  for full projection 
eng. probe_regularization = 0.1;      % Weight factor for the probe update (inertia)

% ADVANCED OPTIONS                     See for more details: Odstr?il M, et al., Optics express. 2018 Feb 5;26(3):3108-23.
% position refinement 
eng. apply_subpix_shift = true;       % apply FFT-based subpixel shift, it is automatically allowed for position refinement
eng. probe_position_search = N_pos_corr;      % iteration number from which the engine will reconstruct probe positions, from iteration == probe_position_search, assume they have to match geometry model with error less than probe_position_error_max
eng. probe_geometry_model = {'scale', 'asymmetry', 'rotation', 'shear'};  % list of free parameters in the geometry model, choose from: {'scale', 'asymmetry', 'rotation', 'shear'}
%eng. probe_geometry_model = {};  % list of free parameters in the geometry model, choose from: {'scale', 'asymmetry', 'rotation', 'shear'}
eng. probe_position_error_max = inf; % maximal expected random position errors, probe prositions are confined in a circle with radius defined by probe_position_error_max and with center defined by original positions scaled by probe_geometry_model
eng. apply_relaxed_position_constraint = false; % added by YJ. Apply a relaxed constraint to probe positions. default = true. Set to false if there are big jumps in positions.
eng. update_pos_weight_every = 100; % added by YJ. Allow position weight to be updated multiple times. default = inf: only update once.

% multilayer extension 
eng. delta_z = [];                     % if not empty, use multilayer ptycho extension , see ML_MS code for example of use, [] == common single layer ptychography , note that delta_z provides only relative propagation distance from the previous layer, ie delta_z can be either positive or negative. If preshift_ML_probe == false, the first layer is defined by position of initial probe plane. It is useful to use eng.momentum for convergence acceleration 
eng. regularize_layers = 0;            % multilayer extension: 0<R<<1 -> apply regularization on the reconstructed object layers, 0 == no regularization, 0.01 == weak regularization that will slowly symmetrize information content between layers 
eng. preshift_ML_probe = true;         % multilayer extension: if true, assume that the provided probe is reconstructed in center of the sample and the layers are centered around this position 

% other extensions 
eng. background = 0;                   % average background scattering level, for OMNI values around 0.3 for 100ms, for flOMNI <0.1 per 100ms exposure, see for more details: Odstrcil, M., et al., Optics letters 40.23 (2015): 5574-5577.
eng. background_width = inf;           % width of the background function in pixels,  inf == flat background, background function is then convolved with the average diffraction pattern in order to account for beam diversion 
eng. clean_residua = false;            % remove phase residua from reconstruction by iterative unwrapping, it will result in low spatial freq. artefacts -> object can be used as an residua-free initial guess for netx engine

% wavefront & camera geometry refinement     See for more details: Odstrcil M, et al., Optics express. 2018 Feb 5;26(3):3108-23.
eng. probe_fourier_shift_search = inf; % iteration number from which the engine will: refine farfield position of the beam (ie angle) from iteration == probe_fourier_shift_search
eng. estimate_NF_distance = inf;       % iteration number from which the engine will: try to estimate the nearfield propagation distance using gradient descent optimization  
eng. detector_rotation_search = inf;   % iteration number from which the engine will: search for optimal detector rotation, preferably use with option mirror_scan = true , rotation of the detector axis with respect to the sample axis, similar as rotation option in the position refinement geometry model but works also for 0/180deg rotation shared scans 
eng. detector_scale_search = inf;      % iteration number from which the engine will: refine pixel scale of the detector, can be used to refine propagation distance in ptycho 
eng. variable_probe = variable_probe_modes>0;           % Use SVD to account for variable illumination during a single (coupled) scan, see for more details:  Odstrcil, M. et al. Optics express 24.8 (2016): 8360-8369.
eng. variable_probe_modes = variable_probe_modes;         % OPRP settings , number of SVD modes using to describe the probe evolution. 
eng. variable_probe_smooth = 0;        % OPRP settings , enforce of smooth evolution of the OPRP modes -> N is order of polynomial fit used for smoothing, 0 == do not apply any smoothing. Smoothing is useful if only a smooth drift is assumed during the ptycho acquisition 
eng. variable_intensity = variable_probe_modes>0;       % account to changes in probe intensity

% extra analysis
eng. get_fsc_score = false;            % measure evolution of the Fourier ring correlation during convergence 
eng. mirror_objects = false;           % mirror objects, useful for 0/180deg scan sharing -> geometry refinement for tomography, works only if 2 scans are provided 

% custom data adjustments, useful for offaxis ptychography
eng.auto_center_data = false;           % autoestimate the center of mass from data and shift the diffraction patterns so that the average center of mass corresponds to center of mass of the provided probe 
eng.auto_center_probe = false;          % center the probe position in real space before reconstruction is started 
eng.custom_data_flip = [0,0,0];         % apply custom flip of the data [fliplr, flipud, transpose]  - can be used for quick testing of reconstruction with various flips or for reflection ptychography 
eng.apply_tilted_plane_correction = ''; % if any(p.sample_rotation_angles([1,2]) ~= 0),  this option will apply tilted plane correction. (a) 'diffraction' apply correction into the data, note that it is valid only for 'low NA' illumination  Gardner, D. et al., Optics express 20.17 (2012): 19050-19059. (b) 'propagation' - use tilted plane propagation, (c) '' - will not apply any correction 

% I/O
eng.plot_results_every = Niter_plot_results;
eng.save_results_every = Niter_save_results;
eng.save_images ={'obj_ph','probe_mag','probe'};
eng.extraPrintInfo = data_descriptor;

resultDir = strcat(p.base_path, sprintf(p.scan.format, p.scan_number),'/roi',p.scan.roi_label,'/');
[eng.fout, p.suffix] = generateResultDir(eng, resultDir);

%add engine
[p, ~] = core.append_engine(p, eng);    % Adds this engine to the reconstruction process

%% Step 10: Run the reconstruction
tic
out = core.ptycho_recons(p);
toc

