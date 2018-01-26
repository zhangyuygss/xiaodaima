%% fpenalty: Penalty function
function [y] = fpenalty(x)
	Alpha = 1e-5;
	left = Alpha*sum((x-1).^2);
	right = (sum(x.^2) - 1/4)^2;
	y = left + right;
end
