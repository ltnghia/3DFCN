function [text_in, text_out] = vid2feat (dir_img, dir_gt, dir_input, dir_output, dir_txt_input, dir_txt_output, param)

%clear
%clc

%dir_img = '/mnt/sda/ltnghia/Dataset/DAVIS2017_TrainVal/JPEGImages/480p/bear';
%dir_gt = '/mnt/sda/ltnghia/Dataset/DAVIS2017_TrainVal/Annotations/480p/bear';
%dir_input = 'temp/input/bear';
%dir_output = 'temp/output/bear';
%dir_txt_input = 'temp/txt_in/bear';
%dir_txt_output = 'temp/txt_out/bear';

%param.block_length = 4;
%param.block_stride = 2;
%param.height = 112;
%param.width = 112;
%param.total_classes = 2;

%==========================================================================

text_in = {};
text_out = {};

if(~exist(dir_input, 'dir'))
    mkdir(dir_input);
end

if(~exist(dir_output, 'dir'))
    mkdir(dir_output);
end

if(~exist(dir_txt_input, 'dir'))
    mkdir(dir_txt_input);
end

if(~exist(dir_txt_output, 'dir'))
    mkdir(dir_txt_output);
end

block_length = param.block_length;
block_stride = param.block_stride;
step = param.step;
new_h = param.height;
new_w = param.width;
total_classes = param.total_classes;
if(total_classes==2) 
    type = 'saliency';
else
    type = 'semantic';
end

im = dir(dir_img);
im = im(3:end);
gt = dir([dir_gt '/*.png']);

zero_img = zeros(new_h, new_w);

for k=1:step
    dir_input0 = [dir_input '/' num2str(k)];
    if(~exist(dir_input0, 'dir'))
        mkdir(dir_input0);
    end
    dir_output0 = [dir_output '/' num2str(k)];
    if(~exist(dir_output0, 'dir'))
        mkdir(dir_output0);
    end
    
    fin = fopen([dir_txt_input '/' num2str(k) '.txt'], 'w');
    fout = fopen([dir_txt_output '/' num2str(k) '.txt'], 'w');
    
    for i=(1-(k-1)*block_stride):block_length:length(im)
        ims = {};
        gts = {};
%         for j=1:block_length 
%             idx = i + j - 1;
%             if(idx < 1 || idx > length(im))
%                 ims = [ims; zero_img];
%                 gts = [gts; zero_img];
%                 disp('zero');
%             else
%                 disp([dir_img '/' im(idx).name]);
%                 ims = [ims; imread([dir_img '/' im(idx).name])];
%                 gts = [gts; imread([dir_gt '/' gt(idx).name])];
%             end
%         end
        for j=1:block_length 
            idx = i + j - 1;
            if(idx < 1)
                idx = 1;
            end
            if(idx > length(im))
                idx = length(im);
            end
            disp([dir_img '/' im(idx).name]);
            ims = [ims; imread([dir_img '/' im(idx).name])];
            try
                gt = imread([dir_gt '/' gt(idx).name]);
            catch
                gt = zero_img;
            end
            gts = [gts; gt];
        end
        feat = imgs2feat(ims, gts, new_h, new_w, type, total_classes);
        txt_in = [dir_input0 '/' sprintf('%05d.bin', i)];
        txt_out = [dir_output0 '/' sprintf('%05d', i)];
        write_4d_feat(feat, txt_in, false);
        fprintf(fin, '%s 0\n', txt_in);
        fprintf(fout, '%s\n', txt_out);
        text_in = [text_in; txt_in];
        text_out = [text_out; txt_out];
        
        disp('---');
        
        %feat_4d = read_4d_feat(txt_in, false);
        %feat_3d = convert_4d_to_3d(feat_4d);
        %figure
        %imshow(uint8(feat_3d{2}(:,:,1:3)))
        %figure
        %imshow(double(feat_3d{2}(:,:,4)))
    end
    disp('=====');
    fclose(fin);
    fclose(fout);
end

end