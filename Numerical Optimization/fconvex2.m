%% fconvex2: convex 2 function
function [y] = fconvex2(x)
	l = length(x);
	tmp = zeros(l, 1);
	for iter = 1:l
		tmp(iter) = (exp(x(iter)) - x(iter))*iter/10;
	end
	y = sum(tmp);
end
