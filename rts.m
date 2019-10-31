function[x_s,P_s,lik_s] = rts(x_sm,x_p,u,P_sm,P_p,A,B,Q)
    x_t1T      = x_sm; 
    x_tt       = x_p;  % posterior estimate from Kalman filter
    x_t1t      = A*x_p + B*u; % prior estimate from Kalman filter
    P_t1T(:,:) = P_sm;
    P_tt (:,:) = P_p;  % posterior covariance from KF
    P_t1t(:,:) = A*P_p*A' + Q; %  prior covariance from KF
    S_t = P_tt*A'*pinv(P_t1t); % tb denotes time t=t-1
    x_s = x_tt + S_t*(x_t1T - x_t1t);
    P_s = P_tt + S_t*(P_t1T - P_t1t)*S_t'';
    
    % Calculate the likelihood for IMM smoothing
    d_y = x_t1T - x_t1t;
    S_new = (abs(2*pi*P_t1t));
    den = sqrt(det(S_new));
    num = exp(-0.5*d_y'*pinv(P_t1t)*d_y);
    lik_s   = num/den;

