%   LOAD_PROJECTIONS load reconstructed projections from disk to RAM 
%
%  [stack_object, theta,num_proj, par]  =  load_projections(par, exclude_scans, dims_ob, theta, custom_preprocess_fun)
%
%   Inputs: 
%       **par - parameter structure 
%       **exclude_scans - list of scans to be excluded from loading, [] = none
%       **dims_ob  - dimension of the object 
%       **theta - angles of the scans 
%       **custom_preprocess_fun - function to be applied on the loaded data, eg cropping , rotation, etc 
%
%   *returns* 
%       ++stack_object - loaded complex-valued projections 
%       ++theta - angles corresponding to the loaded projections, angles for missing projections are removed 
%       ++num_proj - number of projections 
%       ++par - updated parameter structure 

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
%   in written form in the publication: "Data processing was carried out 
%   using the "cSAXS matlab package" developed by the CXS group,
%   Paul Scherrer Institut, Switzerland." 
%   Variations on the latter text can be incorporated upon discussion with 
%   the CXS group if needed to more specifically reflect the use of the package 
%   for the published work.
%
% A publication that focuses on describing features, or parameters, that
%    are already existing in the code should be first discussed with the
%    authors.
%   
% This code and subroutines are part of a continuous development, they 
%    are provided "as they are" without guarantees or liability on part
%    of PSI or the authors. It is the user responsibility to ensure its 
%    proper use and the correctness of the results.



function [stack_object, theta,num_proj, par]  = load_projections(par, exclude_scans, dims_ob, theta, custom_preprocess_fun)

import ptycho.* 
import utils.* 
import io.*
import plotting.*

if nargin < 5
    custom_preprocess_fun = []; 
end
if ~isempty(custom_preprocess_fun)  && ishandle(custom_preprocess_fun) && ~strcmpi(func2str(custom_preprocess_fun), '@(x)x')
    custom_preprocess_fun = [] ;
end

scanstomo = par.scanstomo; 

% avoid loading scans listed in 'exclude_scans'
if  ~isempty(exclude_scans)
    ind = ismember(scanstomo, exclude_scans); 
    scanstomo(ind) = []; 
    theta(ind) = []; 
end
    


% % plot average vibrations for each of the laoded projections 
% disp('Checking stability of the projections')
% poor_projections = prepare.plot_sample_stability(par, scanstomo, ~par.online_tomo, par.pixel_size); 
% if sum(poor_projections) && ...
%         (par.online_tomo || ~strcmpi(input(sprintf('Remove %i low stability projections: [Y/n]\n',sum(poor_projections)), 's'), 'n') )
%     theta(poor_projections) = []; 
%     scanstomo(poor_projections) = []; 
% else
%     disp('All projections are fine')
% end



verbose(1,'Checking available files')
missing_scans = []; 
for num = 1:length(scanstomo)
    progressbar(num, length(scanstomo))
    proj_file_names{num} = find_ptycho_filename(par.analysis_path,scanstomo(num),par.fileprefix,par.filesuffix, par.file_extension);
    if isempty(proj_file_names{num})
        missing_scans(end+1) = scanstomo(num); 
    end
end

verbose(par.verbose_level); % return to original settings

figure(1)
subplot(2,1,1)
hold on 
plot(missing_scans, theta(ismember(scanstomo, missing_scans)), 'rx')
hold off
legend({'Measured angles', 'Missing projections'})
axis tight


if ~isempty(missing_scans)
    ind = ismember(scanstomo, missing_scans); 
    verbose(1,['Scans  not found are ' num2str(missing_scans)])
    verbose(1,['Projections not found are ' num2str(find(ind))])
    scanstomo(ind) = []; 
    theta(ind) = []; 
    proj_file_names(ind) = []; 
else
    verbose(1,'All projections found')
end

num_proj = length(scanstomo); 

if isfield(par, 'fp16_precision') && par.fp16_precision
    % use uint32 to store half floar precision data
    stack_object=zeros(dims_ob(1),dims_ob(2),num_proj, 'like', fp16.set(1i));
else
    stack_object=zeros(dims_ob(1),dims_ob(2),num_proj, 'like', single(1i));
end
pixel_scale =zeros(num_proj,2);
energy = zeros(num_proj,1); 


tic

if num_proj == 0
    verbose(0, 'No new projections loaded')
    return
end
    

which_missing = false(1,num_proj);     % Include here INDEX numbers that you want to exclude (bad reconstructions)
%{
%% prepare parpool 
% pool = gcp('nocreate'); 
% if isempty(pool) || pool.NumWorkers < par.Nworkers
%     delete(pool);
%     pool = parpool(par.Nworkers);
% end
% pool.IdleTimeout = 600; % set idle timeout to 10 hours
%  
% load at least 10 frames per worker to use well the resources 
block_size = max(1, par.Nworkers)*50; 


%% load data, use parfor but process blockwise to avoid lare memory use 
for block_id = 1:ceil(num_proj/block_size)
    block_inds = 1+(block_id-1)*block_size: min(num_proj, block_id*block_size); 
    verbose(1,'=====   Block %i / %i started ===== ', block_id, ceil(num_proj/block_size))
    utils.check_available_memory
    stack_object_block = zeros(dims_ob(1),dims_ob(2),length(block_inds), 'like', stack_object);
    
    share_mem = shm(true);
    share_mem.allocate(stack_object_block); 
    share_mem.detach(); 
   
    
% ticBytes(gcp);

%% start a smaller block in parallel 
%  parfor(num = block_inds,par.Nworkers)
% if parfor fails, try normal loop 
for num = block_inds
      
    file = proj_file_names{num};
        
    if ismember(scanstomo(num), exclude_scans)
        warning(['Skipping by user request: ' file{1}])
        continue  % skip the frames that are listed in exclude_scans
    end
    
    if ~iscell(file)
        file = {file};  % make them all cells 
    end
    
    object= []; 
    for jj = length(file):-1:1
        disp(['Reading file: ' file{jj}])
        % if more than one file is present, try to load the first last one that
        % does not fail 
        try
            object = load_ptycho_recons(file{jj}, 'object');
            object = single(object.object); 
            object = prod(object,4);   % use only the eDOF object if multiple layers are available
            pixel_scale(num,:) = io.HDF.hdf5_load(file{jj}, '/reconstruction/p/dx_spec'); 
            energy(num) = io.HDF.hdf5_load(file{jj}, '/reconstruction/p/energy'); 

            break
        end  
    end

    if isempty(object) || all(object(:) == 0 )
        which_missing(num) = true;
        warning(['Loading failed: ' [file{:}]])
        continue
    end
    
    if ~isempty(custom_preprocess_fun) 
        object = custom_preprocess_fun(object); 
    end
    
    nx = dims_ob(2);
    ny = dims_ob(1);


    if size(object,2) > nx       
        object = object(:,1:nx);       
    elseif size(object,2) < nx
        object = padarray(object,[0 nx-size(object,2)],'post');
    end
    if size(object,1) > ny
        if par.auto_alignment|| par.get_auto_calibration
            object = object(1:ny,:);
        else
            shifty = floor((size(object,1)-ny)/2);
            object = object([1:ny]+shifty,:);
        end
    elseif size(object,1) < ny
        if par.auto_alignment||par.get_auto_calibration
            object = padarray(object,[ny-size(object,1) 0],'post');
        else
            shifty = (ny-size(object,1))/2;
            object = padarray(object,[ny-size(object,1)-floor(shifty) 0],'post');
            object = padarray(object,[floor(shifty) 0],'pre');
        end
    end

    % if par.showrecons
    %     mag=a+bs(object);
    %     phase=angle(object);
    %     figure(1); clf
    %     imagesc(mag); axis xy equal tight ; colormap bone(256); colorbar; 
    %     title(['object magnitude S',sprintf('%05d',ii),', Projection ' ,sprintf('%03d',num) , ', Theta = ' sprintf('%.2f',theta(num)), ' degrees']);drawnow;
    %     set(gcf,'Outerposition',[601 424 600 600])
    %     figure(2); imagesc(phase); axis xy equal tight; colormap bone(256); colorbar; 
    %     title(['object phase S',sprintf('%05d',ii),', Projection ' ,sprintf('%03d',num) , ', Theta = ' sprintf('%.2f',theta(num)), ' degrees']);drawnow;
    %     set(gcf,'Outerposition',[1 424 600 600])    %[left, bottom, width, height
    %     figure(3); %  imagesc3D(probe); 
    %     axis xy equal tight
    %     set(gcf,'Outerposition',[600 49 375 375])    %[left, bottom, width, height
    %     figure(4); 
    %     if isfield(p, 'err')
    %         loglog(p.err);
    %     elseif isfield(p, 'mlerror') 
    %         loglog(p.mlerror)
    %     elseif isfield(p, 'error_metric')
    %         loglog(p.error_metric(2).iteration,p.error_metric(2).value)
    %     end
    %     title(sprintf('Error %03d',num))
    %     set(gcf,'Outerposition',[1 49 600 375])    %[left, bottom, width, height
    %     drawnow;
    % end

    if isfield(par, 'fp16_precision') && par.fp16_precision
        % convert data to fp16 precision 
        object = fp16.set(object); 
    end
    
%     keyboard
    
    % write loaded object to a small block of shared memory, avoid using
    % parpool data transfer
    share_mem_tmp = share_mem;
    [share_mem_tmp, share_mem_object] = share_mem_tmp.attach(); 
    tomo.set_to_array(share_mem_object, object, num - block_inds(1)); 
    share_mem_tmp.detach();

end  % enf of parfor

% tocBytes(gcp);

tic
verbose(1,'Writting to shared stack_object')
[share_mem, stack_object_block] = share_mem.attach(); 
% write loaded block to the full array, avoid memory reallocation
tomo.set_to_array(stack_object, stack_object_block, block_inds-1);
share_mem.free();
toc


end
%}
verbose(1, 'Data loaded')



verbose(1, 'Find residua')
[Nx, Ny, Nprojections] = size(stack_object); 

object_ROI = {ceil(1+par.asize(1)/2:Nx-par.asize(1)/2),ceil(1+par.asize(2)/2:Ny-par.asize(2)/2)}; 
residua = tomo.block_fun(@(x)(squeeze(math.sum2(abs(utils.findresidues(x))>0.1))),stack_object, struct('ROI', {object_ROI})); 

max_residua = 100; 
poor_projections = (residua(:)' > max_residua) & ~par.is_laminography ;   % ignore in the case of laminography 

if any(poor_projections) 
    verbose(1, 'Found %i/%i projections with more than %i residues ', sum(poor_projections), Nprojections, max_residua)
end


if any(which_missing & ~ismember(scanstomo,  exclude_scans) )
    missing = find(which_missing & ~ismember(scanstomo,  exclude_scans)); 
    verbose(1,['Projections not found are ' num2str(missing)])
    verbose(1,['Scans not found are       ' num2str(scanstomo(missing))])
else
    verbose(1,'All projections loaded')
end
toc

% avoid also empty projections
which_wrong =  poor_projections | squeeze(math.sum2(stack_object)==0)'; 

if any(which_wrong & ~ismember(scanstomo,  exclude_scans) )
    wrong = find(which_wrong & ~ismember(scanstomo,  exclude_scans)); 
    verbose(1,['Projections failed are ' num2str(wrong)])
    verbose(1,['Scans failed are       ' num2str(scanstomo(wrong))])
else
    verbose(1,'All loaded projections are OK')
end


%%% Getting rid of missing projections %%%
which_remove = which_missing | which_wrong; 
if any(which_remove)
    if par.online_tomo || ~strcmpi(input(sprintf('Do you want remove %i missing/wrong projections and keep going (Y/n)?',sum(which_remove)),'s'),'n') 
        disp('Removing missing/wrong projections. stack_object, scanstomo, theta and num_proj are modified')
        
        stack_object(:,:,which_remove) = [];
        scanstomo(which_remove)=[];
        theta(which_remove)=[];
        pixel_scale(which_remove,:) = []; 
        energy(which_remove,:) = []; 

        disp('Done')
    else
        disp('Keeping empty spaces for missing projections. Problems are expected if you continue.')
    end
end


par.scanstomo = scanstomo; 
par.num_proj=numel(scanstomo);

pixel_scale = pixel_scale ./ mean(pixel_scale); 

assert(par.num_proj > 0, 'No projections loaded')


if all(all(abs(pixel_scale)-1 < 1e-6)) || ~any(isfinite(mean(pixel_scale)))
    %if all datasets have the same pixel scale 
    pixel_scale = [1,1]; 
else
    warning('Datasets do not have equal pixel sizes, auto-rescaling projections')
    % use FFT base rescaling -> apply illumination function first to remove
    % effect of the noise out of the reconstruction region
    rot_fun = @(x,sx,sy)(utils.imrescale_frft(x .*  par.illum_sum, sx, sy)) ./ ( max(0,utils.imrescale_frft(par.illum_sum,sx,sy))+1e-2*max(par.illum_sum(:))); 
    stack_object = tomo.block_fun(rot_fun,stack_object, pixel_scale(:,1),pixel_scale(:,2));
    pixel_scale = [1,1]; 
end


par.pixel_scale = pixel_scale; 
par.energy = energy; 

%% clip the projections ampltitude by quantile filter 
if par.clip_amplitude_quantile < 1
    MAX = quantile(reshape(abs(fp16.get(stack_object(1:10:end,1:10:end,:))), [], par.num_proj), par.clip_amplitude_quantile ,1); 
    MAX = reshape(MAX,1,1,par.num_proj); 
    clip_fun = @(x,M)(min(abs(x),M) .* x ./ (abs(x) + 1e-5)); 
    stack_object = tomo.block_fun(clip_fun,stack_object, MAX, struct('use_GPU', true));
end


if size(stack_object,3) ~= length(theta) || length(theta) ~= par.num_proj
    error('Inconsistency between number of angles and projections')
end

if ~isempty(par.tomo_id) && all(par.tomo_id > 0)
    % sanity safety check, all loaded angles correpont to the stored angles
    [~,theta_test] = prepare.load_angles(par, par.scanstomo, [], false);
    if max(abs(theta - theta_test)) > 180/par.num_proj/2
        error('Some angles have angles different from expected')
    end
end




end
