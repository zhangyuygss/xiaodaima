function [alpha, x] = wolfe(f, g, x0, d)
	%% Weak wolf 
	% Code from internet
	rho  = 0.25; sigma = 0.75;
	alpha = 1; a = 0; b = Inf;
    k = 0;
	while(1)
        k = k + 1;
        if (k > 300)
            break;
        end
        if (~mod(k,100))
           disp(['    Wolf iter' num2str(k)]); 
        end
		if ~(feval(f, x0+alpha*d) <= feval(f, x0) + rho*alpha*feval(g, x0)'*d)
			b  = alpha;
			alpha = (alpha+a)/2;
			continue;
		end
		if ~(feval(g, x0+alpha*d)'*d >= sigma*feval(g, x0)'*d)
			a = alpha;
			alpha = min([2*alpha, (b+alpha)/2]);
        end
		break;
	end
	x = x0 + alpha*d;
end
