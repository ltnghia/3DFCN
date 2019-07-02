function feat = imgs2feat(ims, gts, new_h, new_w, type, total_classes)

    block_length = length(ims);
    
    vol1 = zeros(new_h, new_w, 3, block_length);
    for l=1:block_length
        im = ims{l};
        im = imresize(im, [new_h new_w]);
        if(size(im,3) == 1)
            im = cat(3, im, im, im);
        end
        vol1(:,:,:,l) = im;
    end
    
    vol2 = zeros(new_h, new_w, 1, block_length);
    
    switch type
        case 'saliency'
            for l=1:block_length
                try
                    im = gts{l};
                    im = mat2gray(im(:,:,1));
                catch
                    im = zeros(new_h, new_w);
                end
                im = imresize(im, [new_h new_w]);
                im(im<0.5) = 0;
                im(im>=0.5) = 1;
                
                data = single(im);
                vol2(:,:,:,l) = data;
            end
        case 'semantic'
            for l=1:block_length
                im = gts{l};
                if(max(im(:)) == 255)
                    im = im2single(im);
                    if(size(im,3) > 1)
                        im = im(:,:,1);
                    end
                    im(im>=0.5) = 1;
                    im(im<0.5) = 0;
                end
                im(im==0) = total_classes;
                im = single(im);

                %data = im_integer_downsample(im, new_h, new_w);
                data = matrix_scale(im, new_h, new_w);

                data = data-1;
                assert(max(data(:))<=total_classes-1);
                assert(min(data(:))>=0);

                vol2(:,:,:,l) = data;
            end
    end
    
    feat = zeros(new_h, new_w, size(vol1,3)+size(vol2,3), block_length);
    feat(:,:,1:size(vol1,3),:) = vol1;
    feat(:,:,size(vol1,3)+1:size(feat,3),:) = vol2;
end