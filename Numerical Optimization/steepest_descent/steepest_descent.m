function [x, val, k, plotinfo] = steepest_descent(f, g, x0)
	%% steepest descent method, line search based on Armijo
	% x: result point  val: result value  k: iterations used
	% f: function to be optimized  g: gradient of f  x0: start point
    %     define them in other function file
	maxiter = 5000;
	epsilon = 1e-3;				% accuracy

	rho = 0.5; sigma = 0.4;		% paras for Armijo

	x1s = 0; x2s = 0; ys = 0;

	for iter = 1:maxiter
		f_x0 = feval(f, x0);
		disp(['Iteration ' num2str(iter-1) '. f_value: ' num2str(f_x0) ', current point:']);
		disp(x0');
		% record optimize path
		x1s = [x1s; x0(1)]; x2s = [x2s; x0(2)]; ys = [ys; f_x0];
		grad = feval(g, x0);
		% fval = feval(f, x0);
		if(norm(grad) < epsilon)
			disp('Optimization done!')
			x = x0;	val = feval(f, x0); k = iter;
			break;
		else
			m = 0; mk = 0;
			while(m < 20)	% Amijo search
				if(feval(f, x0 + (rho^m)*(-grad)) < feval(f, x0) + (rho^m)*sigma*grad'*(-grad))
					mk = m; break; 
				end
				m = m + 1;
			end
			if(m >= 20)
				disp('Amijo did not find a satisfied answer!')
				% mk = m;
			end
			x0 = x0 + (rho^mk)*(-grad);

		end
	end
	if(iter >= 5000)
		disp('Optimization stop early, accuracy not satisfied for 5000 iterations!')
		x = x0;	val = feval(f, x0); k = iter;
	end
	plotinfo = [x1s(2:end), x2s(2:end), ys(2:end)];
end

