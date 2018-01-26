% clear;
% % load milk512.mat;
% % im = mat2gray(milk512);
% im = imread('s14.bmp');
% im = double(imresize(im, [128,128]));
% imsize = size(im);
% 
% subplot(1,2,1);  imshow(mat2gray(im));
% title('Original image');
% 
% logim = log10(im);
% 
% [h0, h1, f0, f1] = daub(2);
% n = 4;
% y = wt2d(logim, h0, h1, n);
% y1 = zeros(imsize);
% y1(1:2^(7-n),1:2^(7-n)) = y(1:2^(7-n),1:2^(7-n));
% xh = iwt2d(y1, f0, f1, n);
% logim1 = logim - xh;
% im1 = 10.^(logim1);
% subplot(1,2,2);  imshow(im1);

clear;
x1 = -2 : 0.05 : 2;
x2 = -2 : 0.05 : 2;
y = zeros(length(x1), length(x2));
y1 = y;
for ii = 1:length(x1)
	for jj = 1:length(x2)
		y(ii, jj) = 2*x1(ii)^2 - x2(jj)^2 + x1(ii)-x2(jj);
		y1(ii,jj) = x1(ii)-x2(jj);
	end
end
surfc(x1, x2, y); hold on;
surfc(x1, x2, y1)