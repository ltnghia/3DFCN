function des = matrix_scale(src, h, w)

des = imresize(src, [h w], 'nearest');

end
