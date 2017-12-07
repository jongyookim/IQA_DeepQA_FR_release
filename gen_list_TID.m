clear
fclose all;

%% Parameters
base_path = 'D:/DB/IQA/TID2013/';
out_file = 'TID2013.txt';
% base_path = 'D:/DB/IQA/TID2008/';
% out_file = 'TID2008.txt';
ref_subpath = 'reference_images/';
dist_subpath = 'distorted_images/';

fid = fopen([base_path 'mos_with_names.txt'], 'r');
% image dst_idx dst_type dst_lev dmos_std dmos
formatSpec = '%f %s';
data = textscan(fid, formatSpec, [Inf, 2]);
% data = textscan(fid, formatSpec);
fclose(fid);

scores = data{1};
dist_name = data{2};

% Norm scores
% fprintf('Orignal: %f ~ %f\n', min(scores), max(scores))
% scores = (scores - min(scores)) / (max(scores) - min(scores));
% fprintf('Norm.  : %f ~ %f\n', min(scores), max(scores))

%% Dis/Ref images
n_files = size(dist_name, 1);
dist_imgs = cell(n_files, 1);
ref_imgs = cell(n_files, 1);
ref_idx = zeros(n_files, 1);
dist_idx = zeros(n_files, 1);
for im_idx = 1:n_files
    ref_name = [dist_name{im_idx}(1:3), '.bmp'];
    ref_idx(im_idx) = str2num(ref_name(2:3));
    dist_idx(im_idx) = str2num(dist_name{im_idx}(5:6));
    
    dist_imgs{im_idx} = [dist_subpath dist_name{im_idx}];
    ref_imgs{im_idx} = [ref_subpath ref_name];
end

% MOSs
fprintf('Orignal: %f ~ %f\n', min(scores), max(scores))
scores = scores / 9;
fprintf('Norm.  : %f ~ %f\n', min(scores), max(scores))

%% Write
fid = fopen([base_path out_file], 'w');
for im_idx = 1:n_files
    fprintf(fid, '%d %d %s %s %f\n', ref_idx(im_idx) - 1, dist_idx(im_idx) - 1, ...
        ref_imgs{im_idx}, dist_imgs{im_idx}, scores(im_idx));
end
fclose(fid);
