function write_3d_feat(feat, dir_output)

[h, w, c] = size(feat);
%save([dir_output '.mat'], 'feat');
switch c
    case 1
        imwrite(feat, [dir_output '.png']);
        %imwrite(mat2gray(feat), [dir_output '.png']);
    case 2
        imwrite((feat(:,:,2)), [dir_output '.png']);
        %imwrite(mat2gray(feat(:,:,2)), [dir_output '.png']);
    case 3
        imwrite(mat2gray(feat), [dir_output '.jpg']);
    otherwise
        save([dir_output '.mat'], 'feat');
end

end
