clear
fclose all;

%% Parameters
base_path = 'D:/DB/IQA/LIVE/LIVE IQA DB/';
n_dist_set = [227, 233, 174, 174, 174];
dist_subpath = {'jp2k/', 'jpeg/', 'wn/', 'gblur/', 'fastfading/'};
ref_subpath = 'refimgs/';
ref_name_file = 'refnames_all.mat';
dmos_file = 'dmos_realigned.mat';
out_file = 'LIVE_IQA.txt';
% out_file = 'LIVE_IQA_nonorm.txt';

%% Dis/Ref images
load([base_path, ref_name_file]);
n_files = sum(n_dist_set);
ref_imgs = refnames_all';
for idx = 1:n_files
    ref_imgs{idx} = [ref_subpath, refnames_all{idx}];
end
dist_imgs = cell(n_files, 1);
dist_types = zeros(n_files, 1);
idx = 1;
for dist_idx = 1:5
    for im_idx = 1:n_dist_set(dist_idx)
        dist_imgs{idx} = [dist_subpath{dist_idx}, sprintf('img%d.bmp', im_idx)];
        dist_types(idx) = dist_idx;
        idx = idx + 1;
    end
end

%% Resolutions
res_list = zeros(n_files, 2);
for idx = 1:n_files
    ref_img = imread([base_path ref_imgs{idx}]);
    [height_r, width_r, ch_r] = size(ref_img);
%     dist_img = imread([base_path dist_imgs{idx}]);
%     [height_d, width_d, ch_d] = size(dist_img);
%     if height_r ~= height_d
%         fprintf('Height not matched %s - %s', dist_imgs{idx}, ref_imgs{idx})
%     end
%     if width_r ~= width_d
%         fprintf('Width not matched %s - %s', dist_imgs{idx}, ref_imgs{idx})
%     end
%     if ch_r ~= ch_d
%         fprintf('Channel not matched %s - %s', dist_imgs{idx}, ref_imgs{idx})
%     end
    res_list(idx, :) = [height_r, width_r];
end

%% DMOSs
mos_str = load([base_path, dmos_file]);
dmos_live = mos_str.dmos_new';
% dmos_max = max(dmos_live);
% dmos_min = 0;
% dmos_live(dmos_live < 0) = 0;
% mos_data = (dmos_live - dmos_min) / (dmos_max - dmos_min);
mos_data = dmos_live;

%% Sort
[ref_imgs_, I] = sort(ref_imgs);
dist_types_ = dist_types(I);
dist_imgs_ = dist_imgs(I);
mos_data_ = mos_data(I);
res_list_ = res_list(I, :);

%% Ref idx
ref_idx = zeros(n_files, 1);
ref_cnt = 1;
prev_ref_name = ref_imgs_{1};
for im_idx = 1:n_files
    cur_ref_name = ref_imgs_{im_idx};
    if strcmp(prev_ref_name, cur_ref_name)
        ref_idx(im_idx) = ref_cnt;
    else
        ref_cnt = ref_cnt + 1;
        prev_ref_name = cur_ref_name;
        ref_idx(im_idx) = ref_cnt;
    end    
end

%% Write
fid = fopen([base_path, out_file], 'w');
for im_idx = 1:n_files
    fprintf(fid, '%d %d %s %s %f %d %d\n', ref_idx(im_idx) - 1, dist_types_(im_idx) - 1, ...
        ref_imgs_{im_idx}, dist_imgs_{im_idx}, mos_data_(im_idx), res_list_(im_idx, 2), res_list_(im_idx, 1));
end
fclose(fid);

