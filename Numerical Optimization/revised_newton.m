function [x, vals, k] = revised_newton(f, g, hesse, x0)
	%% Revist newton's method, modify Gk to make it positive semi-definite
	% x: result point  val: result value  k: total iterations
	% f: function to be optimized  g: gradient of f  x0: start point, column vector
	% ifplot: 1, plot figure (only for 2 dim var)  hesse: Hesse matrix of f	
	maxiter = 500;
	epsilon = 1e-5;

	tau = 0;

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
			% revised part
			muk = norm(grad_x0)^(1+tau);
			Ak = h_x0 + muk*eye(length(startp));
			d = -Ak\grad_x0;

			% [mk, x0] = amijo(f, g, x0, d);
			[alpha, x0] = wolfe(f, g, x0, d);
%             x0 = x0 + d;
		end
	end
	x = xs(2:end, :);
	vals = ys(2:end);
	k = iter;
end
