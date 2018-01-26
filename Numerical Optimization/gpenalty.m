%% gpenalty: function description
function [g] = gpenalty(x)
	l = length(x);
	Alpha = 1e-5;
	rightroot = sum(x.^2) - 1/4;
	g = zeros(l, 1);
	for iter = 1:l
		g(iter) = Alpha*2*(x(iter) - 1) + 2*rightroot*2*x(iter);
	end
end
