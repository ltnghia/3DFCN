function write_4d_feat(feat, dir_output, isImage)
    f = fopen(dir_output, 'w');
    [height, width, channels, length] = size(feat);
    s = int32([1 channels length height width]);
    fwrite(f, s, 'int32');
    
    if(isImage)
        type = 'uint8';
        feat = uint8(feat);
    else
        type = 'single';
        feat = single(feat);
    end
    
    for c=1:channels
        for l=1:length
            data = feat(:,:,c,l);
            data = reshape(data',[1 height*width]);
            fwrite(f, data, type);
        end
    end
    
    fclose(f);
end