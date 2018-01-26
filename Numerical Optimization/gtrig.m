%% gtrig: Gradient of Trigonometric function, minimum square representation
function [g] = gtrig(x)
	n = length(x);
	r = zeros(n, 1);				% left func
	inner_sum = sum(cos(x));		% \sum cosx_j
	tmp = [1:n]';
	J = zeros(n, n);				% Jacobi matrix
	for iter = 1:n
		r(iter) = n - inner_sum + iter*(1-cos(x(iter)) - sin(x(iter)));
		gr = sin(x) + tmp.*sin(x) - cos(x);			% gradient of r_i(x)
		J(iter, :) = gr';
	end
	g = J*r;
end
