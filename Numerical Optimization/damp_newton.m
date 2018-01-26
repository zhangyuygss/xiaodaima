function [x, vals, k] = damp_newton(f, g, hesse, x0, ifplot)
	%% Damp newton's method
	% x: result point  val: result value  k: total iterations
	% f: function to be optimized  g: gradient of f  x0: start point, column vector
	% ifplot: 1, plot figure (only for 2 dim var)  hesse: Hesse matrix of f	
	maxiter = 500;
	epsilon = 1e-5;
    startp = x0;
	xs = x0';
	ys = feval(f, x0)';

	for iter = 1:maxiter
		f_x0 = feval(f, x0);
		grad_x0 = feval(g, x0);
		% Record optimization
		if(iter >= 1)
			xs = [xs; x0'];			
			ys = [ys; f_x0];
		end
		disp(['iteration ' num2str(iter-1) '  current f: ' num2str(f_x0)]);
		if(norm(grad_x0) < epsilon)
			disp('Optimization done!')
			break;
		else
			h_x0 = feval(hesse, x0);
			d = -inv(h_x0)*grad_x0;
			[mk, x0] = amijo(f, g, x0, d);
%             x0 = x0 + d;
		end
	end
	x = xs(2:end, :);
	vals = ys(2:end);
	k = iter;

	if(ifplot && length(startp)==2)		% Can only plot 2 dimension x variable figures
		x1 = (x0(1)-2 : 0.1 : x0(1)+2)';
		x2 = (x0(2)-2 : 0.1 : x0(2)+2)';
		y = zeros(length(x1), length(x2));
		for ii = 1:length(x1)
			for jj = 1:length(x2)
				y(ii, jj) = feval(f, [x1(ii), x2(jj)]);
			end
		end
		surfc(x1, x2, y); hold on;
		p = plot3(xs(:,1), xs(:,2), ys, 'r');
		p.Marker = '*';
	end
end
