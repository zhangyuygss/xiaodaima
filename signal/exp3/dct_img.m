clear;
quality = 10;
load peppers512.mat
im = peppers512;
imsize = size(im);
blockindex = imsize/8;
subplot(2,2,1);  imshow(mat2gray(im));
title('Original image')

qualitys = [90, 40, 10];
for ii = 1:length(qualitys)
	recoverdim = zeros(imsize);
	zeroscnt = 0;
	for id1 = 1:blockindex(1)
		for id2 = 1:blockindex(2)
			B = im((id1-1)*8+1 : id1*8 , (id2-1)*8+1 : id2*8);
			S = compress8x8(B, qualitys(ii));
			zeroscnt = zeroscnt + length(find(S==0));
			B1 = decompress8x8(S, qualitys(ii));
			recoverdim((id1-1)*8+1 : id1*8 , (id2-1)*8+1 : id2*8) = B1;
		end
	end
	zerort = zeroscnt/(imsize(1)*imsize(2));
	Psi = 255;
	Sigmasqr = mean(mean((im-recoverdim).^2));
	PSNR = 10*log10(Psi^2/Sigmasqr);
	subplot(2,2,ii+1);  imshow(mat2gray(recoverdim));
	title(['Quality level: ' num2str(qualitys(ii)) ' zeros: ' ...
		num2str(zerort) ' PSNR: ' num2str(PSNR)]);
end

