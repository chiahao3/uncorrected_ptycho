%% TEST TEMPLATE FOR FUNTIONALITY CHECK OF MULTILAYER EXTENSION IN CPU ENGINES 
% 1) call standard template to get fresh settings defaults 
% 2) generate artificial data that should serve as a standart test "sample"
% 3) call GPU engine with different basic functionalities and test if all still works
%   !! THESE TEST ARE ONLY USEFUL TO FIND CRASHES IN THE CODE, QUALITY OF THE RECONSTRUCTION IS NOT EVALUATED !!


%% set shared parameters for all test scripts 
run(fullfile( fileparts(mfilename('test_ML_data')), 'init_test.m'))


%% general settings
p.   artificial_data_file = 'tests/test_ML_data.m';     % artificial data parameters 
p.   asize = [192 192];                              % size of the reconstruction probe 

%% load simulation parameters 
run(fullfile( ptycho_path, p.artificial_data_file))
Nlayers = length(p.simulation.dataset{1}); 

%% ENGINES
% External C++ code
if isunix
    % Please notice that you have to force data preparation (force_prepare_h5_files=true) if you have made any changes to 
    % the already prepared data (fmag, fmask, positions, sharing ...). 
    eng.  name = 'c_solver';
    eng.  method = 'DM+ML'; 
    eng.  number_iterations = 0;              % Total number of iterations
    eng.  opt_iter = 50;                       % Iterations for optimization     
    eng.  probe_regularization = .1;            % Weigth factor for the probe update; 
    eng.  probe_change_start = 1;               % Start updating probe at this iteration number
    eng.  probe_support_radius = 0.8;           % Normalized radius of circular support, = 1 for radius touching the window    
    eng.  pfft_relaxation = .05;                % Relaxation in the Fourier domain projection, = 0  for full projection    
    eng.  background = 0;                       % [PARTIALLY IMPLEMENTED (not fully optimized)] Add background to the ML model in form:  |Psi|^2+B, B is in average counts per frame and pixel
    eng.  probe_support_fft = false;            % [PARTIALLY IMPLEMENTED (not fully optimized)] Apply probe support in Fourier space, ! uses model zoneplate settings to estimate support size 

    eng.delta_z = p.simulation.thickness  / (Nlayers-1) * ones(Nlayers-1,1); 
   
    
    eng.  single_prec = true;                   % single or double precision
    eng.  threads = 20;                         % number of threads for OMP
    eng.  beamline_nodes = [];                  % beamline nodes for the MPI/OMP hybrid, e.g. ['x12sa-cn-2'; 'x12sa-cn-3'];
    eng.  ra_nodes = 0;                         % number of nodes on ra cluster for the MPI/OMP hybrid; set to 0 for current node
    eng.  caller_suffix = '';                   % suffix for the external reconstruction program
    eng.  reconstruction_program = '';          % specify external reconstruction program that overwrites previous settings, e.g. 'OMP_NUM_THREADS=20 ./ptycho_single_OMP';
    eng.  check_cpu_load = true;                % check if specified nodes are already in use (only x12sa). Disable check if you are sure that the nodes are free.
    eng.  initial_conditions_path = '';         % path of the initial conditions file; default if empty (== prepare_data_path)
    eng.  initial_conditions_file = '';    		% Name of the initial conditions file, default if empty. Do not use ~ in the path
    eng.  measurements_file = '';				% Name of the measurements file, default if empty. Do not use ~ in the path
    eng.  solution_file = '';                   % Name of the solution file, default if empty. Do not use ~ in the path
    eng.  force_prepare_h5_files = 1;           % If true before running the C-code the data h5 file is created and the h5 file with initial object and probe too, regardless of whether it exists. It will use the matlab data preparator. 
    [p, ~] = core.append_engine(p, eng);        % Adds this engine to the reconstruction process
end




% 3D Maximum Likelihood (Matlab)             See for more details: Tsai EH, et al, Optics express. 2016 Dec 12;24(25):29089-108.
if 0  % MEX functions not compatible with Matlab 2018
    eng. name = 'ML_MS';
    eng. ms_opt_iter = 100; 
    eng. N_layer = Nlayers;
    eng.delta_z = p.simulation.thickness  / (Nlayers-1) * ones(Nlayers-1,1); 

    eng. ms_init_ob_fraction = [];            % Default: 1/N_layer    
    eng. ms_opt_flags = [1 1 0];              % Optimize [object, probe, and separation (delat_z)]
    eng. ms_opt_z_param = [200 50];           % [Every this iteratsion, run this many iterations to optimize z (likely to converge earlier anyway)]
    eng. ms_grado_roi = [];                   % Consider using scan_roi   

    
    eng. opt_errmetric = 'L1';                % Error metric for max likelihood = 'poisson', 'L1' (approx poisson), 'L2' (uniform gaussian noise)
    eng. opt_ftol = 1e-10;                    % Tolerance on error metric for optimization
    eng. opt_xtol = 1e-7;                     % Tolerance on optimizable parameters
    eng. probe_mask_bool = true;              % If true, impose a support constraint to the probe
    eng. probe_mask_area = .9;                % Area ratio of the mask
    eng. probe_mask_use_auto = false;         % Use autocorrelation for probe_mask (if false: circular circle)
    eng. scale_gradient = false;              % Preconditioning by scaling probe gradient - Reported useful for weak objects
    eng. inv_intensity = false;               % Make error metric insensitive to intensity fluctuations
    eng. use_probe_support = false;           % Use the support on the probe that was used in the difference-map
    eng. reg_mu = 0.01;  %0.01                % Regularization constant ( = 0 for no regularization)
    eng. smooth_gradient = true;              % Sieves preconditioning, =false no smoothing, = true uses Hanning, otherwise specify a small matrix making sure its sum = 1
    [p, ~] = core.append_engine(p, eng);      % Adds this engine to the reconstruction process
end




run(fullfile( fileparts(mfilename('fullpath')), 'run_test.m'))





% Academic License Agreement
%
% Source Code
%
% Introduction 
% •	This license agreement sets forth the terms and conditions under which the PAUL SCHERRER INSTITUT (PSI), CH-5232 Villigen-PSI, Switzerland (hereafter "LICENSOR") 
%   will grant you (hereafter "LICENSEE") a royalty-free, non-exclusive license for academic, non-commercial purposes only (hereafter "LICENSE") to use the cSAXS 
%   ptychography MATLAB package computer software program and associated documentation furnished hereunder (hereafter "PROGRAM").
%
% Terms and Conditions of the LICENSE
% 1.	LICENSOR grants to LICENSEE a royalty-free, non-exclusive license to use the PROGRAM for academic, non-commercial purposes, upon the terms and conditions 
%       hereinafter set out and until termination of this license as set forth below.
% 2.	LICENSEE acknowledges that the PROGRAM is a research tool still in the development stage. The PROGRAM is provided without any related services, improvements 
%       or warranties from LICENSOR and that the LICENSE is entered into in order to enable others to utilize the PROGRAM in their academic activities. It is the 
%       LICENSEE’s responsibility to ensure its proper use and the correctness of the results.”
% 3.	THE PROGRAM IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR 
%       A PARTICULAR PURPOSE AND NONINFRINGEMENT OF ANY PATENTS, COPYRIGHTS, TRADEMARKS OR OTHER RIGHTS. IN NO EVENT SHALL THE LICENSOR, THE AUTHORS OR THE COPYRIGHT 
%       HOLDERS BE LIABLE FOR ANY CLAIM, DIRECT, INDIRECT OR CONSEQUENTIAL DAMAGES OR OTHER LIABILITY ARISING FROM, OUT OF OR IN CONNECTION WITH THE PROGRAM OR THE USE 
%       OF THE PROGRAM OR OTHER DEALINGS IN THE PROGRAM.
% 4.	LICENSEE agrees that it will use the PROGRAM and any modifications, improvements, or derivatives of PROGRAM that LICENSEE may create (collectively, 
%       "IMPROVEMENTS") solely for academic, non-commercial purposes and that any copy of PROGRAM or derivatives thereof shall be distributed only under the same 
%       license as PROGRAM. The terms "academic, non-commercial", as used in this Agreement, mean academic or other scholarly research which (a) is not undertaken for 
%       profit, or (b) is not intended to produce works, services, or data for commercial use, or (c) is neither conducted, nor funded, by a person or an entity engaged 
%       in the commercial use, application or exploitation of works similar to the PROGRAM.
% 5.	LICENSEE agrees that it shall make the following acknowledgement in any publication resulting from the use of the PROGRAM or any translation of the code into 
%       another computing language:
%       "Data processing was carried out using the cSAXS ptychography MATLAB package developed by the Science IT and the coherent X-ray scattering (CXS) groups, Paul 
%       Scherrer Institut, Switzerland."
%
% Additionally, any publication using the package, or any translation of the code into another computing language should cite for difference map:
% P. Thibault, M. Dierolf, A. Menzel, O. Bunk, C. David, F. Pfeiffer, High-resolution scanning X-ray diffraction microscopy, Science 321, 379–382 (2008). 
%   (doi: 10.1126/science.1158573),
% for maximum likelihood:
% P. Thibault and M. Guizar-Sicairos, Maximum-likelihood refinement for coherent diffractive imaging, New J. Phys. 14, 063004 (2012). 
%   (doi: 10.1088/1367-2630/14/6/063004),
% for mixed coherent modes:
% P. Thibault and A. Menzel, Reconstructing state mixtures from diffraction measurements, Nature 494, 68–71 (2013). (doi: 10.1038/nature11806),
% and/or for multislice:
% E. H. R. Tsai, I. Usov, A. Diaz, A. Menzel, and M. Guizar-Sicairos, X-ray ptychography with extended depth of field, Opt. Express 24, 29089–29108 (2016). 
%   (doi: 10.1364/OE.24.029089).
% 6.	Except for the above-mentioned acknowledgment, LICENSEE shall not use the PROGRAM title or the names or logos of LICENSOR, nor any adaptation thereof, nor the 
%       names of any of its employees or laboratories, in any advertising, promotional or sales material without prior written consent obtained from LICENSOR in each case.
% 7.	Ownership of all rights, including copyright in the PROGRAM and in any material associated therewith, shall at all times remain with LICENSOR, and LICENSEE 
%       agrees to preserve same. LICENSEE agrees not to use any portion of the PROGRAM or of any IMPROVEMENTS in any machine-readable form outside the PROGRAM, nor to 
%       make any copies except for its internal use, without prior written consent of LICENSOR. LICENSEE agrees to place the following copyright notice on any such copies: 
%       © All rights reserved. PAUL SCHERRER INSTITUT, Switzerland, Laboratory for Macromolecules and Bioimaging, 2017. 
% 8.	The LICENSE shall not be construed to confer any rights upon LICENSEE by implication or otherwise except as specifically set forth herein.
% 9.	DISCLAIMER: LICENSEE shall be aware that Phase Focus Limited of Sheffield, UK has an international portfolio of patents and pending applications which relate 
%       to ptychography and that the PROGRAM may be capable of being used in circumstances which may fall within the claims of one or more of the Phase Focus patents, 
%       in particular of patent with international application number PCT/GB2005/001464. The LICENSOR explicitly declares not to indemnify the users of the software 
%       in case Phase Focus or any other third party will open a legal action against the LICENSEE due to the use of the program.
% 10.	This Agreement shall be governed by the material laws of Switzerland and any dispute arising out of this Agreement or use of the PROGRAM shall be brought before 
%       the courts of Zürich, Switzerland. 
