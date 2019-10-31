function[M_t1T,M_pre] = autocov(P_p,A,Q,M_pre,iter)
    P_tt (:,:) = P_p;  % posterior covariance from KF
    P_t1t(:,:) = A*P_p*A' + Q; %  prior covariance from KF
    S_t = P_tt*A'*pinv(P_t1t); % tb denotes time t=t-1
   
    % Update cross-covariance
    if iter == 1            % check if it is the first iteration (t = T -1)
        M_t1T = M_pre;      % cross-covariance for time t+1
    else
        M_t1T = M_pre*S_t'; % cross-covariance for time t+1
    end
   
    M_pre = P_tt + S_t*(M_t1T - A*P_tt);   % preliminary calculation for cross-covariance for time t
    
    % Note that M_t1T requres computing of S_1, so the t-th iteration of the
    % smoother returns M for the t+1 time.
    % For t = T-1 no calculation takes place of M_t1T takes place, as the
    % algorithm is initialised outside of this function as follows
    
    % M_TT = (I - P_TTb*C'*pinv(C*P_TTb*C' + R)*C)*F_Tb*P_TbTb,
    % where I is a unit matrix, Tb denotes t = T - 1, and F_Tb is the state
    % matrix of the final step.

