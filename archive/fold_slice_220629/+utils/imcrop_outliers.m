% IMCROP_OUTLIERS find the largest region of Mask and remove other 
% the nonconnectd regions 
% 
% mask_new = imcrop_outliers(mask, number_of_objects)
%
% Inputs: 
%  **mask                 binary 2D mask to be parsed 
%  **number_of_objects    number of largest objects to be kept, default = 1
%
%  returns: 
%  ++mask_new             mask after removing all smaller nonconnected objects

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




function mask_new = imcrop_outliers(mask, number_of_objects)
    if nargin == 1
        number_of_objects = 1;
    end

    L0 = double(labelmatrix(bwconncomp(mask)));
    [m,n] = hist(L0(L0>0),unique(L0(L0>0)));
    [~,ind] = sort(m);
    
    mask_new = ismember(L0, n(ind(max(1,end - number_of_objects+1):end)));

end