%% conjugate_gradient: conjugate_gradient optimization
function [xs, vals, k] = conjugate_gradient(f, g, x0, method)
	% xs:Points searched  vals:Values searched  k:Totle iterations
	% f: function to be optimized  g: gradient of f  x0: start point, column vector
	% method:'PRP', 'FR', 'PRP+', 'CD', 'DY'
	maxiter = 1000;
	epsilon = 1e-10;			% Accuracy

	xs = x0';
	vals = feval(f, x0);

	if(norm(feval(g, x0)) < epsilon)
		k = 0;  error('Already in the best point!')
	end
	% Iteration one
	xk = x0;
	gk = feval(g, xk);
	dk =  -gk;
	[alpha, xk1] = wolfe(f, g, xk, dk);

	gk1 = gk;
	for iter = 1:maxiter
		disp(['conjugate_gradient iteration ' num2str(iter)]);
		xs = [xs; xk1'];  vals = [vals; feval(f, xk1)];

		gk = gk1;
		gk1 = feval(g, xk1);
		if(norm(gk1) < epsilon)
			k = iter;
			disp('Optimization done!');  return;
		else
			if(strcmp(method, 'PRP'))
				Beta = gk1'*(gk1 - gk)/(gk'*gk);
			elseif (strcmp(method, 'FR'))
				Beta = gk1'*gk1/(gk'*gk);
			elseif (strcmp(method, 'PRP+'))
				Beta = gk1'*gk1/(gk'*gk);
				Beta = max(Beta, 0);
			elseif (strcmp(method, 'CD'))
				Beta = -gk1'*gk1/(dk'*gk);
			elseif (strcmp(method, 'DY'))
				Beta = gk1'*gk1/(dk'*(gk1 - gk));
			else
				error('Invalid method!')
			end
			dk = -gk1 + Beta*dk;					% Update direction
			[alpha, xk1] = wolfe(f, g, xk1, dk);	% Line search
		end
	end
	k = iter;
	if(k>=maxiter)
		disp(['Can not find answer in ' num2str(maxiter) ' iterations!']);
	end
end

