function [X_resize,map] = rawread(filename,x,y,z,s_x,s_y,s_z,pooling)
% RAWREAD Read a raw file.
% RAWREAD('imagefile.raw', xsize, ysize) reads a "raw" image file
% [X,map] = RAWREAD('imagefile.raw') returns both the image and a
% color map, so that
% [X,map] = rawread('imagefile.raw',sx,sy);
% image(X)
% colormap(map)
% will display the result with the proper colors.
%
% NOTE : map is optional and could be replaced during the display by
% the "colormap('gray')" command
%
% See also IMWRITE, IMREAD, IMAGE, COLORMAP.

dot = max(find(filename == '.'));
suffix = filename(dot+1:dot+3);
if strcmp(suffix,'raw')
   fp = fopen(filename,'rb','b');
   if (fp<0)
      error(['Cannot open ' filename '.']);
   end
   if nargin<2
       if strcmp(suffix,'raw')
           disp('RAW file without size : assume image size is 256x256x256');
           x = 256;
           y = 256;
           z = 256;
       end
   end
   for i = 1:256
       map(i,[1:3]) = [i/256,i/256,i/256];
   end
   [ImageTemp,l] = fread(fp,'uchar');
   if l~=x*y*z
       error(' HSI image file is wrong length')
   end
   X_ImageTemp = reshape(ImageTemp,[x,y,z]);
%    X = X_ImageTemp(1:360,:,:);   % body: adjust first axis as 360
   X = X_ImageTemp(:,:,21:end);  % adjust as needed. 
%    X = X_ImageTemp(:,:,:);  % no sampling
   if pooling == 1
       for j = 1:size(X, 3)
           X_resize_1(:,:,j) = max_pooling(X(:,:,j),s_x,s_y);   
       end
       X_resize_2 = permute(X_resize_1, [3,1,2]);
       for j = 1:(y/s_y)
           X_resize_3(:,:,j) = max_pooling(X_resize_2(:,:,j),s_z,1);   
       end
       X_resize = permute(X_resize_3, [2,3,1]);
   else
       X_resize = X;
   end
   disp(' end');
else
    error(' Image file name must end in ''raw''.');
end
fclose(fp);
end