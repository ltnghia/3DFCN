clear
clc

dataset = 'FBMS';
dir_input = ['../vid2feat/temp/' dataset '/txt/output'];
dir_output = ['../../DSRFCN3D-BMVC2017/' dataset];
dir_img = '/mnt/sda/ltnghia/Dataset/FBMS/all';

output_format = '%04d';
stride = 1;
feat_type = '.sal';

videos = dir(dir_input);
for i=3:length(videos)
    video = videos(i).name;
    file_inputs = dir([dir_input '/' video]);
    file_inputs = file_inputs(3:end);
    input_lists = cell(length(file_inputs), 1);
    for j=1:length(file_inputs)
        file_input = file_inputs(j).name;
        list = importdata([dir_input '/' video '/' file_input]);
        input_lists{j} = list;
    end
    
    output_dir = [dir_output '/' video];
    if(~exist(output_dir, 'dir'))
        mkdir(output_dir);
    end
    
    idx_begin = 1;
    idx_end = length(dir([dir_img '/' video '/*.jpg']));
    
    feat2vid(input_lists, output_dir, [video '_' output_format], feat_type, idx_begin, idx_end, stride);
end

