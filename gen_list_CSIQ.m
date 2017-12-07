clear
fclose all;

%% Parameters
base_path = 'D:/DB/IQA/CSIQ/';
ref_subpath = 'src_imgs/';
dist_subpath = 'dst_imgs/';
out_file = 'CSIQ.txt';

% "csiq_dmos.txt" is made manually by copying the values from "csiq.DMOS.xlsx"
% The contained text is like the following format:
% 1600	1	noise	1	0.061	0.062
% 1600	1	noise	2	0.097	0.206
% 1600	1	noise	3	0.033	0.262
% 1600	1	noise	4	0.107	0.375
% 1600	1	noise	5	0.120	0.467

fid = fopen([base_path 'csiq_dmos.txt'], 'r');
% image dst_idx dst_type dst_lev dmos_std dmos
formatSpec = '%s %d %s %d %f %f';
% data = fscanf(fid, formatSpec, [6 Inf]);
data = textscan(fid, formatSpec);
fclose(fid);

ref_names = data{1};
dist_idx = data{2};
dist_types = data{3};
dist_levs = data{4};
scores = data{6};

% Norm scores
% fprintf('Orignal: %f ~ %f\n', min(scores), max(scores))
% scores = (scores - min(scores)) / (max(scores) - min(scores));
% fprintf('Norm.  : %f ~ %f\n', min(scores), max(scores))

%% Dis/Ref images
n_files = size(ref_names, 1);
dist_imgs = cell(n_files, 1);
ref_imgs = cell(n_files, 1);

for im_idx = 1:n_files
    dist_imgs{im_idx} = [dist_subpath dist_types{im_idx} '/' ...
        ref_names{im_idx} '.' dist_types{im_idx} '.' num2str(dist_levs(im_idx)) '.png'];
    ref_imgs{im_idx} = [ref_subpath ref_names{im_idx} '.png'];
end

%% Ref idx
ref_idx = zeros(n_files, 1);
ref_cnt = 1;
prev_ref_name = ref_names{1};
for im_idx = 1:n_files
    cur_ref_name = ref_names{im_idx};
    if strcmp(prev_ref_name, cur_ref_name)
        ref_idx(im_idx) = ref_cnt;
    else
        ref_cnt = ref_cnt + 1;
        prev_ref_name = cur_ref_name;
        ref_idx(im_idx) = ref_cnt;
    end    
end

%% Write
fid = fopen([base_path out_file], 'w');
for im_idx = 1:n_files
    fprintf(fid, '%d %d %s %s %f\n', ref_idx(im_idx) - 1, dist_idx(im_idx) - 1, ...
        ref_imgs{im_idx}, dist_imgs{im_idx}, scores(im_idx));
end
fclose(fid);
