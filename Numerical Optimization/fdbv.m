%% fdbv: Discrete boundary value
function [fx] = fdbv(x)
	m = length(x);
	fx = 0;
	for iter = 1:m
		fx = fx + r_i(iter, x)^2;
	end
end

function ri = r_i(ii, x)
	n = length(x);
	x(n+1) = 0;
	h = 1/(n+1);
	ti = ii*h;
	if(ii == 1)		% x(0)=0
		ri = 2*x(ii) - x(ii+1) + h^2*(x(ii)+ti+1)^3/2;
	else
		ri = 2*x(ii) - x(ii-1) - x(ii+1) + h^2*(x(ii)+ti+1)^3/2;
	end
end
