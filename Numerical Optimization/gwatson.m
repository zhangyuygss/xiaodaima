%% gwatson: gradient of watson function
function [grad] = gwatson(x)
% x is n dim vector, grad is n dim value
	n = length(x);
	grad = zeros(n, 1);

	% compute sum in the square
	insum = zeros(29, 1);
	for ii = 1:29
		ti = ii/29;
		for jj = 1:n
			insum(ii) = insum(ii) + x(jj)*ti^(jj-1);
		end
	end

	% Compute r_i
	r = zeros(31, 1);
	for ii = 1:29
		r(ii) = r_i(ii, x, n);
	end
	r(30) = x(1);  r(31) = x(2) - x(1)^2 - 1;

	for jj = 1:n
		dfdxj = 0;
		for ii = 1:29
			ti = ii/29;
			dridxj = ((jj-1)*ti^(jj-2) - 2*insum(ii)*ti^(jj-1))*2*r(ii);
			dfdxj = dfdxj + dridxj;
		end
		grad(jj) = dfdxj + 2*r(30);
	end
	grad(1) = grad(1) - 2*x(1)*2*r(31);
	grad(2) = grad(2) + 2*r(31);
end

function ri = r_i(i, x, n)
% compute r_i(x)
	ti = i/29;
	tmp_i1 = 0;  tmp_i2 = 0;
	for jj = 1:n
		tmp_i1 = tmp_i1 + (jj-1)*x(jj)*ti^(jj-2);
		tmp_i2 = tmp_i2 + x(jj)*ti^(jj-1);
	end
	ri = tmp_i1 - tmp_i2^2 -1;
end

