function [dst_img] = max_pooling(img,win_size_X,win_size_Y)
% ��Ӱ��img����ȫ�����ػ�������Ĭ�ϴ���Ϊ�����Σ��߳�Ϊwin_size
% ���سػ����Ӱ��
fun = @(block_struct) max(block_struct.data(:));
X=win_size_X; 
Y=win_size_Y; %window sizes
dst_img = blockproc(img, [X Y], fun);
end

