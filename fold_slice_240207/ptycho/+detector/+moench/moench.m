%MOENCH settings for Moench detector
%   This file is called if you specify p.detector = 'moench'.
%
% ** p          p structure
% returns:
% ++ det        detector structure; later accessible via p.detectors(ii).params
%
% see also: detector.load_detector

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

function [ det ] = moench( p )

local_path = fileparts(mfilename('fullpath')); 

if ~isfield(p, 'beamline')
    p.beamline = 'cSAXS';
end

%% parameter
det.pixel_size = 6.25e-6;                 % detector pixel size
det.file_extension = 'tiff';              % raw data file extension 
det.mask = [local_path, '/moench_valid_mask_nb4.mat'];    % detector mask
det.basepath_dir = 'moench/images/';            % raw data directory
det.mask_saturated_value = 150;          % mask pixels above given value
det.mask_below_value = 0;              % mask pixels below given value
det.data_stored = true;                % false == data are generated "onfly", no need to load / store
det.geometry.sz = [1600 1600];           % detector readout size
det.geometry.mask = [];                 % if the mask is larger than the readout 
                                        % geometry (det.geometry.sz), geometry.mask defines the readout for the mask
                                        % e.g. geometry.mask = {[50:500], [50:500]}
                                        
det.detposmotor = {'hy'; 'hx'};       % cSAXS hexapod motors

det.filt_pinhole = [];%28e-6.*160/det.pixel_size;
det.noint = false;

         
det.orientation = [0 1 1];    % [<Transpose> <FlipLR> <FlipUD>]      

% additional arguments for image_read
det.image_read_extraargs = {};

%% define directory tree structure
% define a function handle for compiling the scan directory tree. Input
% should be the scan number.
det.read_path = @utils.compile_x12sa_dirname; 

%% filename
% specify funcion for compiling the filename. Input and output should be p and a 
% temporary structure, containing information about the currently prepared scan. 
% Cf. '+scans/get_filenames_cSAXS.m' 

det.get_filename = @scans.get_filenames_cSAXS;

% additional parameters for the filename function (det.get_filename)



%cSAXS settings
det.read_path_format = 'S%05d-%05d/S%05d/';
det.filename_pattern{1}.str = '%05d';
det.filename_pattern{1}.del = '_';
det.filename_pattern{1}.content = 'scan';
det.filename_pattern{1}.start = 0;
det.filename_pattern{2}.str = '%05d';
det.filename_pattern{2}.del = '_';
det.filename_pattern{2}.content = 'pos';
det.filename_pattern{2}.start = 0;
if det.noint
    det.filename_pattern{3}.str = 'nb4';
    det.filename_pattern{3}.del = '_';
    det.filename_pattern{3}.content = 'suffix';
    det.filename_pattern{3}.start = 0;
    det.filename_pattern{4}.str = 'noint';
    det.filename_pattern{4}.del = '.';
    det.filename_pattern{4}.content = 'suffix';
    det.filename_pattern{4}.start = 0;
else
    det.filename_pattern{3}.str = 'nb4';
    det.filename_pattern{3}.del = '.';
    det.filename_pattern{3}.content = 'suffix';
    det.filename_pattern{3}.start = 0;
end

if strcmpi(p.beamline, 'TOMCAT')
    % TOMCAT settings
    det.basepath_dir = 'moench/';            % raw data directory

    det.orientation = [1 1 0];    % [<Transpose> <FlipLR> <FlipUD>]
    det.image_read_extraargs{2} = det.orientation;

    det.detposmotor = {'ES2laser1y'; 'ES2laser1z'}; % TOMCAT motors @ endstation 2

    det.read_path_format = 'S%05d-%05d/S%05d/';
    det.filename_pattern{1}.str = '%05d';
    det.filename_pattern{1}.del = '_';
    det.filename_pattern{1}.content = 'scan';
    det.filename_pattern{1}.start = 0;
    det.filename_pattern{2}.str = '%05d';
    det.filename_pattern{2}.del = '_';
    det.filename_pattern{2}.content = 'pos';
    det.filename_pattern{2}.start = 0;
    det.filename_pattern{3}.str = 'd0';
    det.filename_pattern{3}.del = '_';
    det.filename_pattern{3}.content = 'suffix';
    det.filename_pattern{3}.start = 0;
    det.filename_pattern{4}.str = 'f000000000000';
    det.filename_pattern{4}.del = '_';
    det.filename_pattern{4}.content = 'suffix';
    det.filename_pattern{4}.start = 0;
    det.filename_pattern{5}.str = '0';
    det.filename_pattern{5}.del = '_';
    det.filename_pattern{5}.content = 'suffix';
    det.filename_pattern{5}.start = 0;
    det.filename_pattern{6}.str = 'nb4';
    det.filename_pattern{6}.del = '.';
    det.filename_pattern{6}.content = 'suffix';
    det.filename_pattern{6}.start = 0;
    
end


end

