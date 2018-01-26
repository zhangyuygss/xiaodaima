%% gnde: Gradient of loss function of neural network method to solve differential equation
function grad = gnde(p)
	m = 10;					% points selected in interval [1,2]
	xs = [1:1/(m-1):2]';
	grad = zeros(length(p), 1);
	for iter = 1:length(xs)
		gradtmp = ginner(xs(iter), p);
		grad = grad + gradtmp;
	end
end

function [gx] = ginner(x, p)
	n = length(p)/3;
	v = p(1:n);
	theta = p(n+1: 2*n);
	omega = p(2*n+1 : end);	

	fx = finner(x, p);				% 1 dim	

	z       = x*omega - theta;
	d_v     = 2*sigmoid(z) + x*omega.*exp(-z).*(sigmoid(z).^2);
	d_theta = (x*v.*omega - 2*v).*exp(-z).*(sigmoid(z).^2) - ...
			  2*x*v.*omega.*exp(-2*z).*(sigmoid(z).^3);
	d_omega = (3*x*v - x^2*v.*omega).*exp(-z).*(sigmoid(z).^2) + ...
			  2*x^2*v.*omega.*exp(-2*z).*(sigmoid(z).^3);

	gx = [d_v; d_theta; d_omega];
	gx = 2*fx*gx;
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

