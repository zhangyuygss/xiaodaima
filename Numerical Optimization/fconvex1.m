%% fconvex1: function description
function [y] = fconvex1(x)
	tmp = exp(x) - x;
	y = sum(tmp);
end
