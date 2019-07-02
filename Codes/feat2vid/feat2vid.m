function feat2vid(input_lists, output_dir, output_format, feat_type, idx_begin, idx_end, stride, isImage)

    if(nargin < 8)
        isImage = false;
    end
    
    temp_dir = [pwd '/temp'];
    if(exist(temp_dir, 'dir'))
        rmdir(temp_dir, 's');
    end
    
    if(exist(output_dir, 'dir'))
        rmdir(output_dir, 's');
    end
    mkdir(output_dir);

    for i=1:length(input_lists)
        padding = (i-1) * stride;
        
        output_temp_dir = [temp_dir '/' num2str(i)];
        mkdir(output_temp_dir);
        
        for j=1:length(input_lists{i})
            disp([input_lists{i}{j} feat_type]);
            feat_4d = read_4d_feat([input_lists{i}{j} feat_type], isImage);
            feat_3d = convert_4d_to_3d(feat_4d);
            
            for k=1:length(feat_3d)
                if (k + (j-1)*length(feat_3d) + idx_begin - 1 - padding <= idx_end)
                    idx = k + (j-1)*length(feat_3d) + idx_begin - 1 - padding;
                    if(idx >= idx_begin)
                        write_3d_feat(feat_3d{k}, sprintf([output_temp_dir '/' output_format], idx));
                    end
                else
                    break
                end
            end
        end
    end
    
    for idx=idx_begin:idx_end
        im = 0;
        count = 0;
        for i=1:length(input_lists)
            output_temp_dir = [temp_dir '/' num2str(i)];
            dir_im = sprintf([output_temp_dir '/' output_format '.png'], idx);
            if(exist(dir_im, 'file'))
                im = im + im2double(imread(dir_im));
                count = count+1;
            end
        end
        if(count > 0)
            im = im / count;
            im = sigmf(im, [5 0.8]);
            if(length(unique(im)) == 1)
                im(:) = 0;
            else
                im = mat2gray(im);
            end
            imwrite(im, sprintf([output_dir '/' output_format '.png'], idx));
        end
    end
end