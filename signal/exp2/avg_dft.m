clear;
load xn.mat;
FS = 128;
% s1 = xn(1:128);
% s2 = xn(1:256);
% s3 = xn(1:512);
% s4 = xn(1:1024);
% s5 = xn(1:1792);
% S1 = fft(s1);
% S2 = fft(s2);
% S3 = fft(s3);
% S4 = fft(s4);
% S5 = fft(s5);
% subplot(2,3,1)
% plot(abs(S1))
% subplot(2,3,2)
% plot(abs(S2))
% subplot(2,3,3)
% plot(abs(S3))
% subplot(2,3,4)
% plot(abs(S4))
% subplot(2,3,5)
% plot(abs(S5))

% L = 13;
% K = 128;
% s = zeros(L, K);
% S = s;
% for iter = 1:L
% 	s(iter, :) = xn((iter-1)*K + 1 : iter*K);
% 	S(iter, :) = fft(s(iter, :));
% end
% rst = zeros(1, K);
% for iter = 1:K
% 	rst(iter) = sum(S(:, iter))/L;
% end
% figure;
% f = FS*(0:(K/2))/K;
% amp = abs(rst(1:K/2+1));
% plot(f,amp);
% xlabel('f(Hz)'); ylabel('|F(f)|');

for ii = 2:2:6
    L = 13;
    K = 124+ii;
	s = zeros(L, K);
	S = s;
	for iter = 1:L
		s(iter, :) = xn((iter-1)*K + 1 : iter*K);
		S(iter, :) = fft(s(iter, :));
	end
	rst = zeros(1, K);
	for iter = 1:K
		rst(iter) = sum(S(:, iter))/L;
    end
	f = FS*(0:(K/2))/K;
	amp = abs(rst(1:K/2+1));
	subplot(1,3,ii/2);
	plot(f,amp);
	xlabel('f(Hz)'); ylabel('|F(f)|'); title(['K = ' num2str(K)]);
end