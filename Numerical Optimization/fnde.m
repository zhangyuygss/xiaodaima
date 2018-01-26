function f = fnde(p)
	%% Loss function of neural network method to solve differential equation
	% a 1-n-1 3 layer neural network
	% p is 3n dimensional parameter column vector
	m = 10;					% points selected in interval [1,2]
	xs = [1:1/(m-1):2]';
	f = 0;
	for iter = 1:length(xs)
		ftmp = finner(xs(iter), p);
		f = f + ftmp^2;
	end
end

function fx = finner(x, p)
	% x 1 dimensional number
	% p is 3n dimensional parameter column vector
	% f is 1 dim number
	n = length(p)/3;
	v = p(1:n);
	theta = p(n+1: 2*n);
	omega = p(2*n+1 : end);

	z = x*omega - theta;
	fx = 2*v'*sigmoid(z) + x*(v.*omega)'*(exp(-z).*(sigmoid(z).^2)) - x^3 + 2/5/x;
end
