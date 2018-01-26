%% fitR: function description
function [r] = fitR(x)
	y = [0.1957, 0.1947, 0.1735, 0.1600, 0.0844,...
		0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246]';
	t = [4, 2, 1, 0.5, 0.25, 0.1670,...
		0.1250, 0.1, 0.0833, 0.0714, 0.0625]';
	r = y - (x(1)*(t.^2 + x(2)*t))./(t.^2 + x(3)*t + x(4));
end