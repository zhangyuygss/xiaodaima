clear;
N = 49; M = (N-1)/2;

n = 1:M-1;
hd = (sin(0.3*pi*(n-24)) + 0.5*sin((n-24)*pi) - 0.5*sin(0.6*pi*(n-24)))./(pi*(n-24));
hd = [hd, 0.5, fliplr(hd)];

% Kaiser window
beta = 3.68;
w = besseli(0, beta*sqrt(1-((n/M) - 1).^2))/besseli(0, beta);
w = [w, 1, fliplr(w)];

h = hd.*w;

[H, ww] = freqz(h, 1, 1024);
plot(ww, 20*log10(abs(H)));
