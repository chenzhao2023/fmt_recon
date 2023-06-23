function [dst_img] = max_pooling(img,win_size_X,win_size_Y)
% 对影像img进行全局最大池化操作，默认窗口为正方形，边长为win_size
% 返回池化后的影像
fun = @(block_struct) max(block_struct.data(:));
X=win_size_X; 
Y=win_size_Y; %window sizes
dst_img = blockproc(img, [X Y], fun);
end

