function [psys, x_tt, mu_tt] = brpf(dyn, y_t, theta, psys, dynamics, phi, t)
% Input:
%   dyn      - dynamics
%   y_t      - observation
%   u_t      - control  input
%   phi      - mode transition probabilities
%   dynamics - dynamical matrices
%   psys     - particle set:
%       psys.nx    - length of the state vector
%       psys.nm    - number of modes
%       psys.np    - number of particles
%     arrays:
%       psys.x_1t  - full set of filtered estimates until t-1, size {nx}(t-1 x np)
%       psys.x_T   - full set of smoothed estimates until t-1, size {nx}(t-1 x np)

%       psys.mu_tt - set of mode probabilities, size {nm}(t x np)
%       psys.x_p   - particles at time t-1, size (nx x np)
%       psys.mu_p  - particles at time t-1, size (nm x np)
%       psys.w_p   - particles weights at time t-1, size (np x 1)
%       psys.weights - particles weights, size (np x t)

%     functions:
%       psys.proposal    - proposal distribution is a Gaussian mixture:
             % means     - a matrix with n-by-k elements. each mean is a row.
             % sigmas    - a matrix with n-by-n-by-k elements. each n-by-n matrix is a covariance
             % mus       - a vector with 1-by-k elements. of weights
%       psys.pdf_x_mid_x - state update pdf, size (psys.nm x 1)
%       psys.pdf_y_mid_x - observation pdf, size  (psys.nm x 1)

% Output:
% x_tt  - evaluated state
% mu_tt - evaluated mode probabilities
% psys  - updated particle set

%% Resampling, get psys_tilde
psys_tilde = psys;
% standard multinomial weighted sample
index      = randsample(1:psys.np, psys.np, true, psys.w_p);
% resample continuous states
psys_tilde.x_p(:,:)  = psys.x_p(:,index);
% resample mode probabilities
psys_tilde.mu_p(:,:) = psys.mu_p(:,index);
% equalise weights
psys_tilde.w_p(:,1)  = 1/psys_tilde.np;

% resample particles at all previous times - equivalent to smoothing
for it = 1:t-1
    for ix = 1:psys.nx
        temp1 =  psys.x_T{ix}(it,:);
        psys.x_T{ix}(it,:) = temp1(:,index);
    end
%     for im = 1:psys.nm
%         temp2 =  psys.mu_tt{im}(it,:);
%         psys.mu_tt{im}(it,:) = temp2(:,index);
%     end
end
%% HMM filter
% Prior mode probabilities
for i=1:psys.np
    for l=1:psys.nm
        sum = 0;
        for k=1:psys.nm
            sum = sum + phi(k,l)*psys_tilde.mu_p(l,i);
        end 
        mu_tilde(l,i) = sum;
    end
end
%% Bayesian filter
% Prior states
for i=1:psys.np
    means{i} = zeros(psys.nx,psys.nm);
    for k=1:psys.nm
         means{i}(:,k) = dynamics{k}(psys_tilde.x_p(:,i),theta);
    end
    psys.x_p(:,i) = random(psys.proposal(means{i}',mu_tilde(:,i)'))';
   
end
% save state particles to memory
for ix = 1:psys.nx
    psys.x_1t{ix} = [psys.x_1t{ix};psys.x_p(ix,:)];
    psys.x_T{ix} = [psys.x_T{ix};psys.x_p(ix,:)];
end
% Joint pdfs
for i=1:psys.np
    sum = 0;
    for l=1:psys.nm
       gamma{i}(l) = psys.pdf_y_mid_x{l}(y_t,psys.x_p(:,i)) * psys.pdf_x_mid_x{l}(psys.x_p(:,i),psys_tilde.x_p(:,i),theta) * mu_tilde(l,i); 
       sum = sum + gamma{i}(l);
    end
    gamma_sum{i} = sum;
end
% Posterior mode probabilities
for i=1:psys.np
    for l=1:psys.nm
        psys.mu_p(l,i) = gamma{i}(l)/gamma_sum{i};
    end
end
% save mode probability particles to memory
for l=1:psys.nm
    psys.mu_tt{l} = [psys.mu_tt{l};psys.mu_p(l,:)];
end
        
%% Updating weights
% Compute
sum = 0;
for i=1:psys.np
    weight(i) = gamma_sum{i}*psys_tilde.w_p(i,1)/pdf(psys.proposal(means{i}',mu_tilde(:,i)'),psys.x_p(:,i)');
    sum = sum +  weight(i);
end
weights_sum = sum;
% Normalise
for i=1:psys.np
    psys.w_p(i) = weight(i)/weights_sum;
end
% save weights to memory
psys.weights = [psys.weights,psys.w_p];

%% Compute central estimates
x_tt    = zeros(psys.nx,1);
mu_tt   = zeros(psys.nm,1);

x_tt = psys.x_p*psys.w_p;
mu_tt = psys.mu_p*psys.w_p;

% for i=1:psys.np
%     x_tt  = x_tt + psys.w_p(i)*psys.x_p(:,i);
%     mu_tt = mu_tt + psys.w_p(i)*psys.mu_p(:,i);
% end

end % of the function


