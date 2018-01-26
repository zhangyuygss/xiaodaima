%% ftrig: Trigonometric function
function [y] = ftrig(x)	
	n = length(x);
	r = zeros(n, 1);				% left func
	inner_sum = sum(cos(x));		% \sum cosx_j
	for iter = 1:n
		r(iter) = n - inner_sum + iter*(1-cos(x(iter)) -sin(x(iter)));
	end
	y = sum(r'*r);
end
