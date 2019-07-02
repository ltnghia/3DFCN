function [feat, blob] = read_4d_feat(dir_input, isImage, max_height, max_width, min_height, min_width)

if(nargin > 1)
    if(isImage)
        precision = 'uint8';
    else
        precision = 'single';
    end
else
    precision = 'single';
end

if(nargin < 3)
    max_height = 112;
    max_width = 112;
    min_height = 112;
    min_width = 112;
end

zero_padding_h = ceil((max_height - min_height) / 2);
zero_padding_w = ceil((max_width - min_width) / 2);

%num x chanel x length x height x width
[s, blob, read_status] = read_binary_blob_preserve_shape(dir_input, precision);

if(nargin > 2)
    
    for n=1:s(1)
        if(max_height > 0 && max_width > 0)
            feat{n} =  zeros(max_height, max_width, s(2), s(3), precision);
        else
            feat{n} =  zeros(s(4), s(5), s(2), s(3), precision);
        end
        
        for c=1:s(2)
            for l=1:s(3)
                im = squeeze(blob(n, c, l, :, :));
                if(max_height > 0 && max_width > 0)
                    if(length(unique(im)) > 2)
                        im = imresize(im, [max_height max_width]);
                    else
                        im = matrix_scale(im, max_height, max_width);
                    end

                    im(1 : zero_padding_h, :) = 0;
                    im(zero_padding_h + min_height + 1 : max_height, :) = 0;
                    im( : , 1 : zero_padding_w) = 0;
                    im( : , zero_padding_w + min_width + 1 : max_width) = 0;
                end
               feat{n}(:, :, c, l) = im;
           end
        end
    end
else
    for n=1:s(1)
        if(max_height > 0 && max_width > 0)
            feat{n} =  zeros(max_height, max_width, s(2), s(3), precision);
        else
            feat{n} =  zeros(s(4), s(5), s(2), s(3), precision);
        end
        for c=1:s(2)
            for l=1:s(3)
               im = blob(n, c, l, :, :);
               %imshow(im);
               feat{n}(:, :, c, l) = im;
           end
        end
    end
end

end

