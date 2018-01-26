clear;
randn('state', 17);
x = randn(1,8);
X = fft(x)
xo = [x(2), x(4), x(6), x(8)];
Xo = fft(xo);

xe = [x(1), x(3), x(5), x(7)];
Xe = fft(xe);

w = exp(-1j*2*pi/8);
W = [1, w, w^2, w^3];
WXo = W.*Xo;

X1 = Xe + WXo
X2 = Xe - WXo


