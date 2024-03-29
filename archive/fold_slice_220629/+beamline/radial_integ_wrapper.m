% (beamline.)radial_integ_wrapper()
% Reads the radial integration filequeue when the filequeue is enabled 
% by _filequeue_on in SPEC, and calls the radial integration script with 
% parameters generated by radial_integration_SAXS_and_WAXS from scan of
% standards.
% This function is called without arguments to run on multiple nodes in
% parallel. 
% Make sure your current matlab directory is Data10/matlab/.
% To change default settings modify the starting lines in function body.
%
% see also: beamline.radial_integ

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

function radial_integ_wrapper()

import utils.verbose

p=struct();
p.queue_path = utils.abspath('../specES1/radial_integ_queue');
p.det_todo = [1 2];
p.recon_latest_first = 0; % =0 from first; =1 from last; =2 random.


% ----- Modify until here -----

finishup = utils.onCleanup(@(x) radial_integ_exit(x), p);
for i=1:numel(p.det_todo)
    mat_todo=[utils.abspath('../analysis/radial_integration_todo') sprintf('/vargin_det%d.mat',p.det_todo(i))];
    try
        m=load(mat_todo);
    catch err
        error(sprintf('Did not find vargin file for detector %d, check current folder is in matlab/, and the radial integration standards are finished.\n',p.det_todo(i)));
    end
    args_todo{i}=m.args;
end


while 1==1
    
    finishup.update(p);
    
    if ~exist(fullfile(p.queue_path,'in_progress'),'dir')
        mkdir(fullfile(p.queue_path,'in_progress'));
    end
    
    if ~exist(fullfile(p.queue_path,'failed'),'dir')
        mkdir(fullfile(p.queue_path,'failed'));
    end
    
    if ~exist(fullfile(p.queue_path,'done'),'dir')
        mkdir(fullfile(p.queue_path,'done'));
    end
    
    fext = 'dat';
    
    status_ok = true;
    verbose(1,['Touching folder and looking for files in the queue in ' p.queue_path]);
    system(sprintf('touch %s',p.queue_path));
    files_recons = dir(fullfile(p.queue_path,'scan*.dat'));
    
    % Found one file to reconstruct
    if ~isempty(files_recons)
        if p.recon_latest_first==0
            p.file_this_recons = files_recons(1).name;
        elseif p.recon_latest_first==1
            p.file_this_recons = files_recons(end).name;
        else
            p.file_this_recons = files_recons(randi([1 numel(files_recons)])).name;
        end
        finishup.update(p);
        verbose(1,['Found file in queue ' fullfile(p.queue_path,p.file_this_recons)]);
        % now move it quickly before someone else will take it
        try
            io.movefile_fast(fullfile(p.queue_path,p.file_this_recons),fullfile(p.queue_path,'in_progress'))
            verbose(1,['Moving file to ' fullfile(p.queue_path,'in_progress')]);
        catch
            verbose(1,['Failed moving file to ' fullfile(p.queue_path,'in_progress')]);
            pause(1);
            status_ok = false;
        end
        
        if status_ok
            % parse the file
            
            fid = fopen(fullfile(p.queue_path,'in_progress',p.file_this_recons),'r');
            
            tline = fgetl(fid);
            while ischar(tline)
                str_parts = strsplit(tline, ' ');
                if numel(str_parts)>1
                    fname = strtrim(str_parts{1});
                    if strcmpi(fname(1:2), 'p.')
                        % found p entry
                        val = [];
                        for ii=2:numel(str_parts)
                            if ~isempty(strtrim(str_parts{ii}))
                                if ~isnan(str2double(str_parts{ii}))
                                    % found number
                                    val = [val, str2double(str_parts{ii})];
                                else
                                    % found char
                                    val = [val, strtrim(str_parts{ii})];
                                end
                            end
                        end
                        p.(fname(3:end)) = val;
                        
                        
                    elseif strcmpi(str_parts{1}, 'samplename')
                        p.samplename = strjoin(strtrim(str_parts(2:end)), '_');
                    end
                end
                tline = fgetl(fid);
            end
            
            fclose(fid);
            finishup.update(p);
            try
                for i=1:numel(p.det_todo)
                    beamline.integrate_range(p.scan_number,p.scan_number,1,args_todo{i});
                end
                verbose(1,['Radial integration of scan ' num2str(p.scan_number) ' finished, moving queue file to ' fullfile(p.queue_path,'done')]);
                file_move_from = fullfile(p.queue_path,'in_progress',p.file_this_recons);
                file_move_to   = fullfile(p.queue_path,'done',p.file_this_recons);
                io.movefile_fast(file_move_from,file_move_to);
            catch err
                try
                    verbose(1,['Error encountered at scan ' num2str(p.scan_number) ', moving queue file to ' fullfile(p.queue_path,'failed')]);
                    file_move_from = fullfile(p.queue_path,'in_progress',p.file_this_recons);
                    file_move_to   = fullfile(p.queue_path,'failed',p.file_this_recons);
                    io.movefile_fast(file_move_from,file_move_to);
                    disp(err);
                catch err
                    verbose(1,['Error with file system delays, skipping.']);
                    disp(err);
                end
            end
        end
        
    else
        verbose(1,'Did not find enough files in queue, pausing 10 seconds.');
        pause(10);
    end
end
end

function radial_integ_exit(p)
    import utils.verbose;
    verbose(1,'Radial integration interrupted');
    if isfile(fullfile(p.queue_path,'in_progress',p.file_this_recons))
        try
            verbose(1,['Moving current queue file back to ' p.queue_path]);
            file_move_from = fullfile(p.queue_path,'in_progress',p.file_this_recons);
            file_move_to   = fullfile(p.queue_path,p.file_this_recons);
            io.movefile_fast(file_move_from,file_move_to);
        catch
            disp(err);
            verbose(1,'File system error, aborting.');
        end
    end
end