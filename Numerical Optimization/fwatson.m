%% Watson function, fix m = 31, n can be modified
function w = fwatson(x)
% x is n dim vector, w is 1 dim value
	m = 31;  n = length(x);
	w = 0;
	for ii = 1:29
		tmp_i1 = 0;
		tmp_i2 = 0;
		ti = ii/29;
		for jj = 1:n
			tmp_j1 = (jj-1)*x(jj)*ti^(jj-2);
			tmp_j2 = x(jj)*ti^(jj-1);
			tmp_i1 = tmp_i1 + tmp_j1;
			tmp_i2 = tmp_i2 + tmp_j2;
		end
		w = w + (tmp_i1 - tmp_i2^2 - 1)^2;
	end
	% add r30 and r31
	w = w + x(1)^2 + (x(2) - x(1)^2 -1)^2;
end
