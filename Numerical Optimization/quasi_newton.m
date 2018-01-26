function [x, vals, k] = quasi_newton(f, g, xk, method)
	%% Quasi-newton's method
	% x: result point  val: result value  k: total iterations
	% f: function to be optimized  g: gradient of f  xk: start point, column vector
	% method 'SR1', 'DFP', 'BFGS'
	maxiter = 400;
	epsilon = 1e-5;
	
	fevlcount = 0;
	x = xk';
	vals = feval(f, xk);
	fevlcount = fevlcount + 1;

	% Iteration 1, steepest descent
	H = eye(length(xk));
	grad_k = feval(g, xk); 				% gk, gradient at xk
	fevlcount = fevlcount + 1;
	d = -H*grad_k;						% d1
	[alpha, xk1] = wolfe(f, g, xk, d);
	grad_k1 = feval(g, xk1);
	fevlcount = fevlcount + 1;
	delta_x = xk1 - xk; 				% s1
	delta_g = grad_k1 - grad_k;			% y1

	for iter = 1:maxiter
		% Record results
        disp(['Quasi_newton iteration ' num2str(iter)])
		x = [x; xk1'];
		vals = [vals; feval(f, xk1)];
		fevlcount = fevlcount + 1;

		if(norm(grad_k1) < epsilon)
			disp('Optimization done!')		
			disp(['fevalcount: ' num2str(fevlcount) ])
			break;
		else
			if (strcmp(method, 'SR1'))
				delt_H = (delta_x - H*delta_g)*(delta_x - H*delta_g)'/((delta_x - H*delta_g)'*delta_g);
				H = H + delt_H;
			elseif (strcmp(method, 'DFP'))
				delt_H = delta_x*delta_x'/(delta_x'*delta_g) - H*(delta_g*delta_g')*H/(delta_g'*H*delta_g);
				H = H + delt_H;
			elseif (strcmp(method, 'BFGS'))
				delt_H = (1 + delta_g'*H*delta_g/(delta_g'*delta_x))*(delta_x*delta_x')/(delta_g'*delta_x) - ...
						 (delta_x*delta_g'*H + H*delta_g*delta_x')/(delta_g'*delta_x);
				H = H + delt_H;
			else
				error('Invalid method!');
			end
			d = -H*grad_k1;
			xk = xk1;
			[alpha, xk1] = wolfe(f, g, xk1, d);

			% Prepare for next iteration
			grad_k  = grad_k1;
			grad_k1 = feval(g, xk1);
			fevlcount = fevlcount + 1;
			delta_x = xk1 - xk;
			delta_g = grad_k1 - grad_k;
		end
    end
    k = iter;
	if(iter >= maxiter)
        disp(['fevalcount: ' num2str(fevlcount) ])
		disp(['Can not find answer in ' num2str(maxiter) ' iterations!'])
	end
end
