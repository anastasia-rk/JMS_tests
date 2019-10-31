function[x_prior,x_posterior,P_prior,P_posterior,lik] = ukf(y,x,P,Q,R,A,B,C,theta,Z,knots,basis_type)
% preliminaries
L       = length(x);  % state vector dimension
M       = length(y);  % measurement vector dimension
alpha   = 0.1;       % scaling constant - from literature (set to larger value for wider distribution spread)
kappa   = 0;          % from literature
beta    = 2;          % optimal in case distribution of x is Gaussian
lambda  = alpha^2*(L + kappa) - L;  % scaling factor for sigma points
c       = L + lambda;               % scaling factor for covariance
x_prior = zeros(L,1);
y_prior = zeros(M,1);
P_prior = zeros(L,L);
P_yy    = zeros(M,M);
P_xy    = zeros(L,M);

% Weights
W_s(1) = lambda/c;
W_c(1) = lambda/c + (1 - alpha^2 + beta);
W_s (2:2*L+1) = 1/(2*c);  % for states
W_c (2:2*L+1) = 1/(2*c);  % for covariances

% Sigma points
% S = sqrt(c)*chol(P);
% X{1} = x;
% for i = 2:L+1
%     X{i} = x + S(i);
% end
% for i=L+2:2*L+1
%     X{i} = x - S(i-L);
% end

S = sqrt(c)*chol(P);
X{1} = x;
xxx = x(:,ones(1,numel(x)));
X2 = [x xxx+S xxx-S]; 
for i = 2:L+1
    X{i} = X2(:,i);
end
for i=L+2:2*L+1
    X{i} = X2(:,i);
end

% Unscented transformation 
for i=1:2*L+1
    Xx{i}     = dynfun (X{i},A,B,theta,Z,knots,basis_type);                 % sigma state propagation
    x_prior   = x_prior + W_s(i)*Xx{i};                                     % prior estimate
    Yy{i}     = measfun(Xx{i},C);                                           % sigma measuremnent
    y_prior   = y_prior + W_s(i)*Yy{i};                                     % measurement of prior
end

for i=1:2*L+1
    P_prior   = P_prior + W_c(i)*(Xx{i} - x_prior)*(Xx{i} - x_prior)' + Q; % prior covariance
    P_yy      = P_yy + W_c(i)*(Yy{i} - y_prior)*(Yy{i} - y_prior)' + R;    % residual covariance
    P_xy      = P_xy + W_c(i)*(Xx{i} - x_prior)*(Yy{i} - y_prior)';        % cross-covariance
end

% Correction
S_inv = pinv(P_yy);
d_y = y - y_prior;

K           = P_xy*pinv(P_yy);           % Kalman gain
x_posterior = x_prior + K*d_y; % State correction
P_posterior = P_prior - K*P_yy*K';       % Covariance correction


% Calculate likelihood for IMM
S_new = (abs(2*pi*P_yy));
den = sqrt(det(S_new));
num = exp(-0.5*d_y'*S_inv*d_y);
lik   = num/den;