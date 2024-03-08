close all
clear variables
clc

cSAXS_matlab_path= 'H:/fold_slice_Lee'; % Set this to your `fold_slice` path
ptycho_matlab_path = strcat(cSAXS_matlab_path, '/ptycho');

addpath(cSAXS_matlab_path);
addpath(ptycho_matlab_path);
addpath(strcat(ptycho_matlab_path, '/utils'));
addpath(strcat(ptycho_matlab_path, '/utils_electron'));

%% Parameter

template = 'p03_runPtycho_worker_template';
kmax_dir= 'kmax4';
%'kmax2', 'kmax4'
main_dir = strcat('data//test_ptycho_recon_', kmax_dir); % Change this to the actual input folder
if strcmp(kmax_dir, 'kmax2')
    r = 2;
elseif strcmp(kmax_dir, 'kmax4')
    r = 1;
end

N_true = 1024; % size of the probe when generating initial guess, around 1024 is a sweet spot between accuracy / artifact for probe with Cs
voltage = 200; %beam voltage in keV

potential_modes= {'static'};
%{'static', 'phonon'};
coherence_modes= {'coherent'};
%{'coherent', 'temporal', 'spatial', 'temporal_spatial'};
alpha_dirs= {'alpha_7.5mrad_'};
%{'alpha_10.5mrad_', 'alpha_23.1mrad_', 'alpha_40.0mrad_'};
cs_dirs= {'Cs_1.0um_'}; 
%'Cs_1.1mm_'};%{'Cs_0.5um_', 'Cs_1.0mm_'};
target_shapes= {'dp_128_'};
%{'dp_256_', 'dp_512_'}; 
detector_blur_stds= {'blur_0px_'};
%{'blur_0.5px_', 'blur_1px_', 'blur_2px_'};
doses= {'dose_1.0e+08ePerAng2'}; 
%{'dose_1.0e+06ePerAng2','dose_1.0e+08ePerAng2', 'dose_1.0e+14ePerAng2'};

custom_positions_source= 'data/intermediate_files/true_position_std0.1.hdf5';
%%

for potential_mode = potential_modes
    for coherence_mode = coherence_modes
        for alpha_dir = alpha_dirs
            for cs_dir = cs_dirs
                for target_shape = target_shapes
                    for blur = detector_blur_stds
                        for dose = doses

                            % Setup input dir
                            mode_dir = strcat(potential_mode, '_', coherence_mode);
                            data_dir = strcat('data3D_200kV_df_10nm_', alpha_dir, cs_dir, target_shape, blur, dose, '/');
                            input_dir = fullfile(main_dir, mode_dir, data_dir);
                            disp(input_dir)
                            
                            % Define regular expressions
                            numeric_regex = '\d+(\.\d+)?';
                            
                            % Extract numeric parts using regular expressions
                            alpha_numeric = cellfun(@(x) str2double(regexp(x, numeric_regex, 'match', 'once')), alpha_dir);
                            cs_numeric = cellfun(@(x) str2double(regexp(x, numeric_regex, 'match', 'once')), cs_dir);
                            target_shapes_numeric = cellfun(@(x) str2double(regexp(x, numeric_regex, 'match', 'once')), target_shape);
                            if strcmp(cs_dir, 'Cs_1.0um_')
                                cs_numeric = cs_numeric * 1e4 ; %Ang
                            elseif strcmp(cs_dir, 'Cs_1.1mm_')
                                cs_numeric = cs_numeric * 1e-3 * 1e10 ; %Ang
                            end
                            
                            base_path = char(input_dir); %base path needed by ptychoshelves
                            
                            % Setup data-dependent param
                            Ndpx= target_shapes_numeric;  % size of cbed
                            voltage= 200; %keV
                            alpha0= alpha_numeric; % semi-convergence angle (mrad)
                            df= 100; % Ang, defocus. Positive is underfocus, which is align with abTEM and Kirkland.
                            Cs= cs_numeric; %Ang

                            % Run ptycho reconstruction
                            run(template)
                         end
                    end
                end
            end
        end
    end
end

