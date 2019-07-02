clear
clc

dir_img = '/mnt/sda/ltnghia/Dataset/FBMS/all';
dir_gt = '/mnt/sda/ltnghia/Dataset/FBMS/all';
dataset = 'FBMS';

param.block_length = 4;
param.block_stride = 1;
param.step = 4;
param.height = 112;
param.width = 112;
param.total_classes = 2;

data_in = {};
data_out = {};

videos = dir(dir_img);
for k=3:length(videos)
    video = videos(k).name;
    dir_img0 = [dir_img '/' video ];
    dir_gt0 = [dir_gt '/' video '/ground-truth'];
    dir_input = [pwd '/temp/' dataset '/feat/input/' video];
    dir_output = [pwd '/temp/' dataset '/feat/output/' video];
    dir_txt_input = [pwd '/temp/' dataset '/txt/input/' video];
    dir_txt_output = [pwd '/temp/' dataset '/txt/output/' video];
    [text_in, text_out] = vid2feat (dir_img0, dir_gt0, dir_input, dir_output, dir_txt_input, dir_txt_output, param);
    
    data_in = [data_in; text_in];
    data_out = [data_out; text_out];
end

fin = fopen(['/mnt/sda/ltnghia/Code/publish/BMVC2017/' dataset '_input.txt'], 'w');
fout = fopen(['/mnt/sda/ltnghia/Code/publish/BMVC2017/' dataset '_output.txt'], 'w');
for i=1:length(data_in)
    fprintf(fin, '%s 0\n', data_in{i});
    fprintf(fout, '%s\n', data_out{i});
end
fclose(fin);
fclose(fout);





