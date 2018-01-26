clear;
h = lowpass_filter(25, 0.25*pi);
[H, ww] = freqz(h, 1, 1024);
plot(ww, 20*log10(abs(H)));
