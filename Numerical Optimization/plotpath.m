function plotpath(f, x, vals)
	%% Plot optimization path on 3d figure, only for 2d x's
	% f is the function to be optimized, x are points the optimization passed  
	% vals are f(x) 
	x1 = (x(end, 1)-2 : 0.1 : x(end, 1)+2);
	x2 = (x(end, 2)-2 : 0.1 : x(end, 2)+2);
	y = zeros(length(x1), length(x2));
	for ii = 1:length(x1)
		for jj = 1:length(x2)
			y(ii, jj) = feval(f, [x1(ii), x2(jj)]');
		end
	end
	surfc(x1, x2, y); hold on;
	p = plot3(x(:,1), x(:,2), vals, 'r');
	p.Marker = '*';
end

