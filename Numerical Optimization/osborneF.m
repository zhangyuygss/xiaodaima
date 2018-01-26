%% Fosborne: function description
function [y] = osborneF(x)
	rx = osborneR(x);
	y = rx'*rx/2;
end
