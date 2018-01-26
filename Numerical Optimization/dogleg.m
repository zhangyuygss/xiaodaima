%% dogleg: Dogleg method for least-square problem
function [xs, vals, k] = dogleg(r, J, x0)
	% r: residual function vector(m x 1) J: Jacobi matrix(m x n)
	% x0: Initial point(n x 1)
	maxiter = 200;
	epsilon = 1e-6;

	yita1 = 0.25;  yita2 = 0.75;		% Trust region bound
	muk = 1;							
	n = length(x0);

	for iter = 1:maxiter
		disp(['Dogleg iteration ' num2str(iter)])
		rx = feval(r, x0);
		fx = rx'*rx/2;
		if(iter == 1) % record results 
			xs = x0';  vals = fx;
		else
			xs = [xs; x0'];				 	
			vals = [vals; fx];
		end		
		jacobi = feval(J, x0);
		gx = jacobi'*rx;
		if(norm(gx) < epsilon)
			k = iter;	disp('Optimization done!');
			return;
		else
			% Dogleg, compute d_k
			d_gn = -inv(jacobi'*jacobi)*gx; % Gauss-newton direction
			if(norm(d_gn) < muk^2)	% d_gn in trust region
				d = d_gn;
			else
				d_sd = -gx;		% Steepest descent direction
				Alpha = norm(d_sd)^2/norm(jacobi*d_sd)^2; % Accurate search
				if (norm(Alpha*d_sd) >= muk^2)	% d_sd out of trust region
					d = muk*d_sd/norm(d_sd);
				else
					% Solve quadratic (ax^2 + bx + c = 0) function for Beta
					a = norm(d_gn - Alpha*d_sd)^2;
					b = 2*Alpha*d_sd'*(d_gn - Alpha*d_sd);
					c = Alpha^2*norm(d_sd)^2 - muk^2;
					Beta = (-b + sqrt(b^2 - 4*a*c))/(2*a);
					d = (1-Beta)*Alpha*d_sd + Beta*d_gn;
				end
			end

			rx1 = feval(r, x0+d);
			fx1 = rx1'*rx1/2;					% f(x+d)
			Gamma = (fx - fx1)/(d'*(muk*d - gx)/2);
			if(Gamma>0 && Gamma < yita1) 	% Modify muk
				muk  = 4*muk;
			elseif (Gamma > yita2 || Gamma <= 0)
				muk = muk/2;
			end
			if(Gamma <= 0)					% Decide x_{k+1}
				x0 = x0;
			else
				x0 = x0 + d;
			end
		end
    end
    k = iter;
end
