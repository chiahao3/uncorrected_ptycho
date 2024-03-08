%HDF5_DSET_EXISTS check if dataset exists in given file
%   file...         h5 file path
%   dset...         dataset name
%   
%   *optional*
%   gpath...        path within the h5 file; default root (/)
%   check_links...  include links; default true
%
%   EXAMPLES:
%       out = io.HDF.hdf5_dset_exists('./recons.h5',
%       'object_phase_unwrapped', '/reconstruction', true);
%

%*-----------------------------------------------------------------------*
%|                                                                       |
%|  Except where otherwise noted, this work is licensed under a          |
%|  Creative Commons Attribution-NonCommercial-ShareAlike 4.0            |
%|  International (CC BY-NC-SA 4.0) license.                             |
%|                                                                       |
%|  Copyright (c) 2017 by Paul Scherrer Institute (http://www.psi.ch)    |
%|                                                                       |
%|       Author: CXS group, PSI                                          |
%*-----------------------------------------------------------------------*
% You may use this code with the following provisions:
%
% If the code is fully or partially redistributed, or rewritten in another
%   computing language this notice should be included in the redistribution.
%
% If this code, or subfunctions or parts of it, is used for research in a 
%   publication or if it is fully or partially rewritten for another 
%   computing language the authors and institution should be acknowledged 
%   in written form in the publication: “Data processing was carried out 
%   using the “cSAXS matlab package” developed by the CXS group,
%   Paul Scherrer Institut, Switzerland.” 
%   Variations on the latter text can be incorporated upon discussion with 
%   the CXS group if needed to more specifically reflect the use of the package 
%   for the published work.
%
% A publication that focuses on describing features, or parameters, that
%    are already existing in the code should be first discussed with the
%    authors.
%   
% This code and subroutines are part of a continuous development, they 
%    are provided “as they are” without guarantees or liability on part
%    of PSI or the authors. It is the user responsibility to ensure its 
%    proper use and the correctness of the results.

function [out] = hdf5_dset_exists(file, dset, varargin)

out = false;

% load info
if nargin > 2
    h = h5info(file, varargin{1});
else
    h = h5info(file);
end

if nargin > 3
    check_links = varargin{2};
else
    check_links = true;
end

if nargin > 4
    check_groups = varargin{3};
else
    check_groups = true;
end

% loop through datasets and check if name exists
for ii=1:numel(h.Datasets)
    if strcmpi(h.Datasets(ii).Name, dset)
        out = true;
        break
    end
end

if check_links
    for ii=1:numel(h.Links)
        if strcmpi(h.Links(ii).Name, dset)
            out = true;
            break
        end
    end
end

if check_groups
    for ii=1:numel(h.Groups)
        [~, gname] = fileparts(h.Groups(ii).Name);
        if strcmpi(gname, dset)
            out = true;
            break
        end
    end
end

end

