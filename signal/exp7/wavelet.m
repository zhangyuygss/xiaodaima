%% Wavelet denoising
function denosimg = wavelet(nosimg, n, thresholding)
	% noiseimg: noised image, gray square image of size 2^k
	% n: 2 for haar filter; 4 for db4; 6 for db6
	% thresholding: 'hyperbolic' for hyperbolic thresholding 
	% 				'soft' for soft thresholding
	addpath(genpath('./wave3'));
	imsize = size(nosimg);
	[h0, h1, f0, f1] = daub(n);
	y = wt2d(nosimg, h0, h1, log2(imsize(1)));
	N = imsize(1);
	d1 = y(N/2+1:N, N/2+1:N);
	db = mean(mean(d1));
	s2 = sum(sum((d1-db).^2))/((N/2)^2-1);
	delta = 2*sqrt(log10(N)*s2);
	if (strcmp(thresholding, 'hyperbolic'))
		yh = sign(y).*sqrt(max(y.^2-delta^2, zeros(N,N)));
	elseif (strcmp(thresholding, 'soft'))
		yh = sign(y).*(max(abs(y)-delta, zeros(N,N)));
	else
		error('Invalid thresholding!');
	end
	xh = iwt2d(yh, f0, f1, log2(N));
	xh = min(ones(imsize), xh);
	xh = max(zeros(imsize), xh);
	denosimg = xh;
end

