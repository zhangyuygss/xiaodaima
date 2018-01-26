%% gconvex2: function description
function [g] = gconvex2(x)
	l = length(x);
	g = zeros(l, 1);
	for iter = 1:l
		g(iter) = (exp(x(iter)) - 1)*iter/10;
	end
end
