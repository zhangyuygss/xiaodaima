%% LMF: LMF least-square optimization
function [xs, vals, k] = LMF(r, J, x0)
	% r: residual function vector(m x 1) J: Jacobi matrix(m x n)
	% x0: Initial point(n x 1)
	maxiter = 200;
	epsilon = 1e-6;

	yita1 = 0.25;  yita2 = 0.75;		% Trust region bound
	muk = 1;							
	n = length(x0);

	for iter = 1:maxiter
		disp(['LMF iteration ' num2str(iter)])
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
			d = -inv(jacobi'*jacobi + muk*eye(n))*jacobi'*rx;
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
