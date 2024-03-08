close all; clear all; clc;

cSAXS_matlab_path= 'H:/fold_slice_Lee'; % Set this to your `fold_slice` path
ptycho_matlab_path = strcat(cSAXS_matlab_path, '/ptycho');

addpath(cSAXS_matlab_path);
addpath(ptycho_matlab_path);
addpath(strcat(ptycho_matlab_path, '/utils'));
addpath(strcat(ptycho_matlab_path, '/utils_electron'));

save_dir = 'data/test_matlab_ssim_lines';
mkdir(save_dir);

base_dir = strcat('data/test_ptycho_recon_kmax4/'); % Where you store the reconstruction data
base_dir2 = 'data/intermediate_files'; % Where you store the ground truth file
ground_truth_path = 'transmission_static_dx0.08Ang.mat';
ground_truth_scaling = 1.5; % The loaded ground truth would be rescale based on this factor. The scaling is needed because of the abTEM and .hdf5 resampling of the CBED. If it's 3, it means the ground truth would be shrunk to 1/3 of the original pixel number (3x larger physical size of the pixel) 

mode_dirs = {'static_coherent'}; % {'static_temporal_spatial'};
alpha_max_s = {'7.5'}; %{'1.0','3.0','5.0','7.5','9.5','10.5','12.0','15.0','18.0','20.0','22.0','23.1','24.0','26.0','28.0','30.0', '32.0','34.0','36.0','38.0','40.0'}; %convergence angle in mrad
Cs_s = {'Cs_1.0um_'}; %{'Cs_1.0um_', 'Cs_1.1mm_'};
N_s = [128]; % [256, 512]; % size of diffraction pattern in pixel. only square dp allowed
postfixs = {'_blur_0px_dose_1.0e+08ePerAng2'}; %{'_blur_0px_dose_1.0e+06ePerAng2', '_blur_0px_dose_1.0e+07ePerAng2', '_blur_0px_dose_1.0e+08ePerAng2'};
dose_s = [1e8]; %[1e6, 1e7, 1e8];

ssims = zeros(length(alpha_max_s),1);


tic

%% parameters
for mm = 1:length(mode_dirs)
    for ii = 1:length(postfixs)
    
        for jj = 1:length(Cs_s)
    
            for kk = 1:length(N_s)
                
                subimages_ph = cell(length(alpha_max_s),1);
                for ll = 1:length(alpha_max_s)
                    
                    mode_dir  = mode_dirs{mm};
                    postfix   = char(postfixs(ii));
                    dose      = dose_s(ii);
                    Cs        = char(Cs_s(jj));
                    N         = N_s(kk);
                    alpha_max = char(alpha_max_s(ll));
                
                    data_dir = strcat('data3D_200kV_df_10nm_alpha_', alpha_max, 'mrad_', Cs, 'dp_', num2str(N), postfix);
                    disp(data_dir)
                    recon_dir = strcat('1\roi_1_Ndp_', num2str(N), '/MLs_L1_p4_g256_dpFlip_T/');
                    %%
                    Niter = 500;
                    
                    addpath ../
                    % clear;
                    % close all;
                    params = [];
                    %params = struct;
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %%% Reconstruction files %%%
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    %% For .mat output
                    base_folder = base_dir;
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %%% Alignment parameters %%%
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    params.verbose_level = 0;   % adjust output level
                    params.plotting = params.verbose_level;        % (3) show everything, (2) show aligned images + FSC, (1) show FSC, (0) none
                    params.remove_ramp = 1;     % Try to remove ramp from whole image before initial alignment
                    params.image_prop = 'phasor'; % = 'complex' or = 'phasor' (phase with unit amplitude) or = 'phase'  (Note: phase should not be used if there is phase wrapping)
                    params.crop = '';
                    % '' for using the default half size of the probe
                    % 'manual' for using GUI to select region. This will display the range, e.g. {600:800, 600:800}
                    % {600:800, 600:800} for custom vertical and horizontal cropping, respectively
                    params.flipped_images = 0; % If images are taken with a horizontal flip, e.g. 0 & 180 for tomography
                    params.GUIguess = 0;       % To click for an initial alignment guess, ignores the values below
                    params.guessx = [];       % Some initial guess for x alignment
                    params.guessy = [];
                    params.electron = true;
                    %%%%%%%%%%%%%%%%%%%%%%
                    %%% FSC parameters %%%
                    %%%%%%%%%%%%%%%%%%%%%%
                    params.taper = 20;             % Pixels of image tapering (smoothing at edges) - Increase until the FSC does not change anymore
                    params.SNRt = 0.5;      % SNRt = 0.2071 for 1/2 bit threshold for resolution of the average of the 2 images
                    % SNRt = 0.5    for 1   bit threshold for resolution of each individual image
                    params.thickring = 3;  % Thickness of Fourier domain ring for FSC in pixels
                    params.freq_thr = 0.05;  % (default 0.05) To ignore the crossings before freq_thr for determining resolution
                    
                    %%%%%%%%%%%%
                    %%% misc %%%
                    %%%%%%%%%%%%
                    params.prop_obj = false;     % propagation distance at the sample plane; leave empty to use the value from the reconstruction p structure; set to "false" for no propagation
                    params.apod = [];            % if true, applies an apodization before propagating by params.prop_obj, the apodization border is around the valid reconstruction region; leave empty to use the value from the reconstruction p structure
                    params.lambda = [];           % wavelength; needed for propagating the object; leave empty to use the value from the reconstruction p structure
                    params.pixel_size = [];       % pixel size at the object plane; leave empty to use the value from the reconstruction p structure
                    params.xlabel_type = 'resolution';
                    %%%%%%%%%%%%%%%%%%%%
                    %%% FP parameter %%%
                    %%%%%%%%%%%%%%%%%%%%
                    
                    %%% the following parameters are ignored, unless p.fourier_ptycho==true %%%
                    params.filter_FFT = true;           % apply a circular mask to the reconstructed spectrum (needs p.plot_maskdim)
                    params.crop_factor = 0.9;           % crop final image by the given factor
                    params.crop_asize = [800 800];      % crop object before applying the FFT
                    params.z_lens = 49.456e-3;          % FZP focal distance
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %%% Do not modify below %%%
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                  %% Compare data1 with the ground truth
                    file = {};
                    for i=1:2
                        if i==1
                            recon_file = fullfile(base_folder, mode_dir, data_dir,strcat(recon_dir,strcat('Niter',num2str(Niter),'.mat')));
                            a = dir(recon_file);
                            
                            % Check if the file exists using the dir function
                            file_info = dir(recon_file);
                            
                            % If the file does not exist or is empty
                            if isempty(file_info) || file_info.bytes == 0
                                error(['Error: The file ', recon_file, ' does not exist or is empty.']);  % Display an error message and terminate
                            end
    
                            load(fullfile(a(1).folder,a(1).name))
                            object1 = imresize(object, 1/1,'bilinear');
                            file{1} = object1;
                            cen = floor(size(object1)/2)+1;
                            params.pixel_size = p.dx_spec;
                            offset = [0,0]; %130;
                            %keep ROI the same for all cases
                            crop_roi{1} = cen(1)-((128/2) - offset(1)):cen(1)+ ((128/2) + offset(1));
                            crop_roi{2} = cen(2)-((128/2) - offset(2)):cen(2)+ ((128/2) + offset(2));
                            params.crop{i} = crop_roi;
                        else
                            recon_file = fullfile(base_dir2, ground_truth_path);
                
                            load(recon_file)
                            object1 = imresize(object, 1/ground_truth_scaling,'bilinear');
                            file{2} = object1;
                            cen = floor(size(object1)/2)+1;
                            params.pixel_size = p.dx_spec;
                            offset = [3,0]; %130;
                            %keep ROI the same for all cases
                            crop_roi{1} = cen(1)-((128/2) - offset(1)):cen(1)+ ((128/2) + offset(1));
                            crop_roi{2} = cen(2)-((128/2) - offset(2)):cen(2)+ ((128/2) + offset(2));
                            params.crop{i} = crop_roi;
                        end
                    end
                    [resolution, stat, subim1, subim2] = aligned_FSC(file{1}, file{2}, params);
                    
                    % Reset subim1_ph and subim2_ph for each iteration
                    subim1_ph = [];  
                    subim2_ph = [];
    
                    subim1_ph = phase_unwrap(angle(subim1));
                    subim2_ph = phase_unwrap(angle(subim2));
                    ssims(ll,1) = ssim(subim2_ph, subim1_ph);
                    subimages_ph{ll,1} = subim1_ph;
                end
                
                %figure; imagesc(subim1_ph); colormap parula; axis image;
                %figure; imagesc(subim2_ph); colormap parula; axis image;
                
                
                save(strcat(save_dir,'/ssim_', mode_dir,'_',Cs, 'dp_', num2str(N), postfix,'.mat'),'ssims','alpha_max_s','dose','Cs','N','base_dir','subimages_ph');
                
                figure;           
                plot(str2double(alpha_max_s), ssims(:,1));
                xlabel('Semi-Convergence Angle (mrad)')
                ylim([0 1]);
                ylabel('SSIM');
                title(strcat(mode_dir,'_',Cs, 'dp_', num2str(N), char(postfix)), 'Interpreter', 'none');
                set(gca,'Fontsize', 18)
    
            end
        end
    end
end
toc

%%



