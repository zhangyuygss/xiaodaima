function [mk, x] = amijo(f, g, x0, d)
	%% Amijo condition  
	% code from book  
	beta = 0.5; sigma = 0.4;
	m = 0; maxiter = 20;
	while(m < maxiter)
		if(feval(f, x0+beta^m*d) <= feval(f, x0) + sigma*beta^m*feval(g, x0)'*d)
			mk = m; x = x0 + beta^mk*d;
			break;
		end
		m = m + 1;
	end
	disp('Amijo can not find a satisfied step length in 20 iterations!')
	mk = m; x = x0 + beta^mk*d;
end
