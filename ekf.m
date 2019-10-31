function[x_posterior,P_posterior,lik] = ekf(y,x,P,Q,R,A,B,C,theta,Z,knots,basis_type)
% approximations of F and H
dx = 0.001;
for i=1:length(x)
xx = x;
xx(i) = x(i) + dx; 
F(:,i) = (dynfun (xx,A,B,theta,Z,knots,basis_type) - dynfun (x,A,B,theta,Z,knots,basis_type))./dx;
H(:,i) = (measfun(xx,C) - measfun(x,C))./dx;
end
% Suboptimal estimator
x_prior     = dynfun(x,A,B,theta,Z,knots,basis_type);
P_prior     = F*P*F' + Q;
d_y          = y - measfun(x,C);
S           = H*P_prior*H' + R;
S_inv       = pinv(S);
K           = P_prior*H'*S_inv;
x_posterior = x_prior + K*d_y;
P_posterior = (eye(length(x)) - K*H)*P_prior;
% Calculate likelihood for IMM
S_new = (abs(2*pi*S));
den = sqrt(det(S_new));
num = exp(-0.5*d_y'*S_inv*d_y);
lik   = num/den;