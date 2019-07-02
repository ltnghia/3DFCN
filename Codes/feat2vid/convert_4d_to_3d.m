function output = convert_4d_to_3d(input)
input = input{1};
[height, width, channels, length] = size(input);

output = {};
for l=1:length
    feat = input(:,:,:,l);
    output{l} = feat;
end

end