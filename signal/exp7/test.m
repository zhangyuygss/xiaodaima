clear;
load milk512.mat;
im = mat2gray(milk512);
imsize = size(im);
% Add noise 
randn('state', 7);
w = 0.2*randn(imsize);
noisedim = im + w;
nf = norm(im, 'fro');
SNR_bf = 20*log10(nf/(norm(noisedim-im, 'fro')));
subplot(1,2,1);  imshow(im);
title('Original image');
subplot(1,2,2);  imshow(noisedim);
title(['Noised image SNR:' num2str(SNR_bf)]);


% Denoising
figure;
haar_soft = wavelet(noisedim, 2, 'soft');
SNR_hs = 20*log10(nf/(norm(haar_soft-im, 'fro')));
subplot(2,3,1);  imshow(haar_soft);  
title(['harr-soft SNR:' num2str(SNR_hs)]);

db4_soft = wavelet(noisedim, 4, 'soft');
SNR_d4s = 20*log10(nf/(norm(db4_soft-im, 'fro')));
subplot(2,3,2);  imshow(db4_soft);  
title(['db4-soft SNR:' num2str(SNR_d4s)]);

db6_soft = wavelet(noisedim, 6, 'soft');
SNR_d6s = 20*log10(nf/(norm(db6_soft-im, 'fro')));
subplot(2,3,3);  imshow(db6_soft);  
title(['db6-soft SNR:' num2str(SNR_d6s)]);

haar_hyper = wavelet(noisedim, 2, 'hyperbolic');
SNR_hh = 20*log10(nf/(norm(haar_hyper-im, 'fro')));
subplot(2,3,4);  imshow(haar_hyper);  
title(['haar-hyper SNR:' num2str(SNR_hh)]);

db4_hyper = wavelet(noisedim, 4, 'hyperbolic');
SNR_d4h = 20*log10(nf/(norm(db4_hyper-im, 'fro')));
subplot(2,3,5);  imshow(db4_hyper);  
title(['db4-hyper SNR:' num2str(SNR_d4h)]);

db6_hyper = wavelet(noisedim, 6, 'hyperbolic');
SNR_d6h = 20*log10(nf/(norm(db6_hyper-im, 'fro')));
subplot(2,3,6);  imshow(db6_hyper);  
title(['db6-hyper SNR:' num2str(SNR_d6h)]);

