%% gdbv: Gradient of discrete boundary value
function [grad] = gdbv(x)
	n = length(x);
	h = 1/(n+1);
	t1 = 1*h;  tn = n*h;
	grad = zeros(n, 1);
	grad(1) = (2+3*h^2*(x(1)+t1+1)/2)*2*r_i(1, x) + (-1)*2*r_i(2, x);
	for iter = 2:n-1
		ti = iter*h;
		grad(iter) = (-1)*2*(r_i(iter+1, x) + r_i(iter-1, x)) +...
					 (2+3*h^2*(x(iter)+ti+1)/2)*2*r_i(iter, x);
	end
	grad(n) = (2+3*h^2*(x(n)+tn+1)/2)*2*r_i(n, x) + (-1)*2*r_i(n-1, x);
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

