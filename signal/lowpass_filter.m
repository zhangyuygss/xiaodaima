9function h = lowpass_filter(N, omegac)
	%% Generate low pass filter 
	% N: filter length  omegac: cutoff freq
	M = (N-1)/2;
	n = 0:M-1;
	hd = sin((n-M)*omegac)./((n-M)*pi);
	hd = [hd omegac/pi fliplr(hd)];

	% Hamming window
	w = 0.54 - 0.46*cos(pi*n/M);
	w = [w 1 fliplr(w)];

	% result filter
	h = hd.*w;
end