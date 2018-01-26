%% osborneG: Gradient of osborne problem
function [g] = osborneG(x)
	jacobi = osborneJ(x);
	rx = osborneR(x);
	g = jacobi'*rx;
end
