clear;
quality = 10;
load boat512.mat
im = boat512;
imsize = size(im);
blockindex = imsize/8;
subplot(2,2,1);  imshow(mat2gray(im));

qualitys = [10, 40, 90];
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

	subplot(2,2,ii+1);  imshow(mat2gray(recoverdim));
end

function [S] = compress8x8(B, quality)
	normB = B - 128;
	C = dct2(normB);
	Q = [16,11,10,16,24,40,51,61;...
		12,12,14,19,26,58,60,55;...
		14,13,16,24,40,57,69,56;...
		14,17,22,29,51,87,80,62;...
		18,22,37,56,68,109,103,77;...
		24,35,55,64,81,104,113,92;...
		49,64,78,87,103,121,120,101;...
		72,92,95,98,112,100,103,99];
	if(quality > 50 && quality < 100)
		tau = (100 - quality)/50;
    elseif (quality<=50 && quality>0)
		tau = 50/quality;
	else
		error('Invalid quality level!');
	end
	Q = tau*Q;
	S = round(C./Q);
end

function B = decompress8x8(S, quality)
	Q = [16,11,10,16,24,40,51,61;...
		12,12,14,19,26,58,60,55;...
		14,13,16,24,40,57,69,56;...
		14,17,22,29,51,87,80,62;...
		18,22,37,56,68,109,103,77;...
		24,35,55,64,81,104,113,92;...
		49,64,78,87,103,121,120,101;...
		72,92,95,98,112,100,103,99];
	if(quality > 50 && quality < 100)
		tau = (100 - quality)/50;
    elseif (quality<=50 && quality>0)
		tau = 50/quality;
	else
		error('Invalid quality level!');
    end
	Q = tau*Q;
	R = Q.*S;
	E = idct2(R);
	B = E + 128;
end
