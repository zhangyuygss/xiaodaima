% Code for Line Search Optimization (0.618 method)
clear;
t = (sqrt(5)-1)/2;		% 0.618
% Initial interval [a,b]
a = 0;
b = 1;
% Accuracy e
e = 0.01;

max_iter = 100;

x1 = a + (1 - t)*(b - a);
x2 = a + t*(b - a);

for iter = 1:max_iter
	disp(['Iter ' num2str(iter) '.  Current accuracy:' num2str(abs(b - a))]);
	if(abs(b - a) < e)
		disp(['Optimization done! Running for ' num2str(iter) ' iters.']);
        disp(['Answer: ' num2str((a + b)/2)]);
		break;			% Stop condition satisfied, break
	else 				
		fx1 = f(x1);
		fx2 = f(x2);
		if(fx1 < fx2)
            b = x2; 
			x2 = x1;
			x1 = a + (1 - t)*(b - a);
		else
			a = x1;
			x1 = x2;
			x2 = a + t*(b - a);
		end
	end
end


%% functionname: function f(x), modify it to your own function f(x)
function [y] = f(x)
	y = 1 - x*exp(-x^2) ;
end
