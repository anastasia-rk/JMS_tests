function[lik_s] = sm_lik(x_sm,x_p,P_p,A,B,Q,theta,Z,knots,basis_type)
% approximations of F
dx = 0.001;
for i=1:length(x_p)
xx = x_p;
xx(i) = x_p(i) + dx; 
F(:,i) = (dynfun (xx,A,B,theta,Z,knots,basis_type) - dynfun (x_p,A,B,theta,Z,knots,basis_type))./dx;
end
    x_t1t      = dynfun(x_p,A,B,theta,Z,knots,basis_type); % prior estimate from Kalman filter
    P_t1t(:,:) = F*P_p*F' + Q; %  prior covariance from KF
    d_y = x_sm - x_t1t;
    S_new = (abs(2*pi*P_t1t));
    den = sqrt(det(S_new));
    num = exp(-0.5*d_y'*pinv(P_t1t)*d_y);
    lik_s   = num/den;