%% gauss_newton: Damped gauss_newton method for least-square
function [xs, vals, k] = gauss_newton(f, g, r, J, x0)
	% f: function to be optimized  g: gradient of f
	% r: residual function vector(m x 1) J: Jacobi matrix(m x n)
	% x0: Initial point(n x 1)
	maxiter = 200;
	epsilon = 1e-6;

	for iter = 1:maxiter
        disp(['Gauss newton iteration ' num2str(iter)])
		rx = feval(r, x0);
		fx = rx'*rx/2;
		if(iter == 1)					% record results
			xs = x0';  vals = fx;
		else
			xs = [xs; x0'];				 	
			vals = [vals; fx];
		end	
		jacobi = feval(J, x0);			% Jacobi martix
		gx = jacobi'*rx;				% Gradient
		if(norm(gx) < epsilon)
			k = iter;
			disp('Optimization done!');
			return;
		else
			d = -inv(jacobi'*jacobi)*gx;
			[alpha, x0] = wolfe(f, g, x0, d);
		end
    end
    k = iter;
    disp(['Can not find answer in ' num2str(maxiter) ' iterations!']);
end
