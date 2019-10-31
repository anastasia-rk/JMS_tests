my_init
  
%% Define field model parameters
fprintf(1,'Setting model parameters... \n')

% True field (with actual theta)
basis_type = 'bspline';
% Set up limits of the grid: x_min,y_min,x_max,y_max
grid_limits = [0, 0, 1000, 1000];

% Set up the basis function grid
nx = 4; ny = 4;
switch basis_type
    case 'gaussian'
        [knots,sigma] = setup_gaussian_support(grid_limits,nx,ny);
        % equal sigmas for isotopic basis functions
        sigma = 80;
        Z = [sigma^2 0; 0 sigma^2];   
        ll = size(knots,2);
    case 'bspline'
        [knots] = setup_spline_support(grid_limits,nx,ny);
        Z = 0;
        ll = size(knots,2)/2;
end
grid_limits1 = grid_limits;
% Theta = initiate_field(0,60,grid_limits1,knots);  % fit splines to a linear surface
% Theta_old = Theta;

 Theta = ones(ll,1);
 Theta(1:4,1) = 10;
 Theta(5:8,1) = 100;
 Theta(9:12,1) = 190;
 Theta(13:16,1) = 290;
 Theta_model = Theta;
 
done = plot_field(Theta,Z,knots,grid_limits,'bspline');
mu_field = 1;

%% Set up JMS
% Preliminaries
dT       = 1; % sampling time
x_length = 4; % size of the state vector
y_length = 2;
I =   eye(2,2);
O = zeros(2,2);
thta = 0.5; % reversion to mean in O-U process 

% Markov chain
n_models = 2;
p_tr        = [0.9 0.1;...
               0.1 0.9];... 
mu_0 = 1/n_models*ones(1,n_models);

% Brownian motoion with friction  (velocity as O-U process)
F_rw = [I dT*I;...
    O   I - thta*dT*I]; 
% For CV
F_cv = [I dT*I;...
    O   I - thta*dT*I];  
% Observation matrix
C_all = [I O];
% Control matrix
B_cv = [O; dT*I];
B_rw = [O; O];
% Disturbance matrix matrix
G_cv = [dT^2*I/2; dT*I];
G_rw = [dT^2*I/2; dT*I];
%% Disturbance matrices
sig_w_1 = 2; % CV disturbance - random component of cell acceleration 
sig_w_2 = 2; % RW  disturbance
% For DRIFT
mm = sig_w_1^2*G_cv*G_cv';
dd = diag(mm);
Q_cv = diag(dd);
sig_w(:,1) = sqrt(dd);
cov_for_gm(:,:,1) = Q_cv;

mm = sig_w_2^2*G_rw*G_rw';  
dd = diag(mm);
Q_rw = diag(dd);
sig_w(:,2) = sqrt(dd);
cov_for_gm(:,:,2) = Q_rw;

% R - measurement noise
sig_r = 1;
R_v = sig_r^2*I;
for k=1:n_models
    sig_v(:,k) = diag(sqrt(R_v));  
end

A{1} = F_cv;
A{2} = F_rw;
B{1} = B_cv;
B{2} = B_rw;
C{1} = C_all;
C{2} = C_all;
Q{1} = Q_cv;
Q{2} = Q_rw;
R{1} = R_v;
R{2} = R_v;
G{1} = G_cv;
G{2} = G_rw;
%% Pdfs and models
pdf_w     = @(w,sig_wk) mvnpdf(w, zeros(x_length,1), sig_wk);
sample_w  = @(w,sig_wk) mvnrnd(zeros(x_length,1), sig_wk);         % sample process noise

pdf_v     = @(v,sig_vk) mvnpdf(v, zeros(y_length,1), sig_vk);
sample_v  = @(v,sig_vk) mvnrnd(zeros(y_length,1), sig_vk);         % sample measurement noise

sample_x0 = @(x) mvnrnd(x, eye(4));               % initial distribution

dyn       = @(A, B, x, theta) A*x + B*field_gradient(x(1:2),Z,knots,basis_type)*theta;     
obs       = @(C, x) C*x;

for k=1:n_models
    dynamics{k} = @(x,theta) dyn(A{k},B{k},x,theta);
    obesrvations{k} = @(x) obs(C{k},x);
end

%% Create a particle system
%   psys - particle set:
%       psys.nx    - length of the state vector
%       psys.nm    - number of modes
%       psys.np    - number of particles
%     arrays:
%       psys.x_1t  - full set of continuous states until t-1, size {np}(nx x t-1)
%       psys.mu_tt - set of mode probabilities, size {np}(nm x 1)
%       psys.x_p   - particles at time t-1, size (nx x np)
%       psys.mu_p  - particles at time t-1, size (nm x np)
%       psys.w_p   - particles weights at time t-1, size (N_p x 1)
%       psys.weights   - particles weights at time t-1, size (N_p x t)


%     functions:
%       psys.proposal    - proposal distribution
%       psys.pdf_x_mid_x - state update pdf, size (psys.nm x 1)
%       psys.pdf_y_mid_x - observation pdf, size  (psys.nm x 1)

psys.nx     = x_length;
psys.nm     = n_models;
psys.np     = 100;
% fill in pdfs
for k=1:psys.nm
    psys.pdf_x_mid_x{k} = @(x_t, x, theta) pdf_w(x_t - dyn(A{k},B{k},x,theta),Q{k});
    psys.pdf_y_mid_x{k} = @(y, x) pdf_v(y   - obs(C{k},x),R{k});
end
% Proposal distribution is a Gaussian mixture
% means     - a matrix with n-by-k elements. each mean is a row.
% sigmas    - a matrix with n-by-n-by-k elements. each n-by-n matrix is a covariance
% mus       - a vector with 1-by-k elements. of weights
psys.proposal = @(means,mus) gmdistribution(means,cov_for_gm,mus);

%% Estimation
models = 1;
for iModel = models
clear X Y Mode_model
load(['simulated_tracks_' num2str(iModel)]);
% Tracks = [1:nTracks]; 
Tracks = 1; % for now
for k=Tracks
    T = length(Y{k});
    for t=1:T
        y{t} = Y{k}(:,t);
    end

    psys.w_p     = 1/psys.np*ones(psys.np,1);
    psys.x_p     = zeros(psys.nx,psys.np);
    psys.weights = psys.w_p;
    for ip=1:psys.np
        psys.x_p(:,ip) = sample_x0([y{1};0;0]);
    end
    psys.mu_p   = 1/psys.nm*ones(psys.nm,psys.np);
    for ix = 1:psys.nx
        psys.x_1t{ix}(1,:) = psys.x_p(ix,:);      
        psys.x_T{ix}(1,:) = psys.x_1t{ix}(1,:);
    end
    for im = 1:psys.nm
        psys.mu_tt{im}(1,:) = 1/psys.nm*ones(1,psys.np);
    end
    cols{1} = (10/max(psys.weights(:,1)))*psys.weights(:,1); %linspace(1,10,psys.np);   
%% recurtion cycle for the RB particle filter
for t=2:T    
    [psys,x_weighted{k}(:,t),mu_weighted{k}(:,t)] = brpf(dyn, y{t}, Theta, psys, dynamics, p_tr, t);
    [max_mode,i] = max(mu_weighted{k}(:,t));
    mode_probable{k}(t) = i;
    cols{t} = (10/max(psys.weights(:,t)))*psys.weights(:,t); %linspace(1,10,psys.np);
    
end
end % for track (k)

end % for model (iModel)

%% 
figure; 
plot(X{k}(1,:),X{k}(2,:),'k','LineWidth',2); hold on;
xlabel('X, a.u.'); ylabel('Y, a.u.');
plot(x_weighted{k}(1,2:t-1),x_weighted{k}(2,2:t-1),'r','LineWidth',2); hold on;
legend('true','filtered')
%% states and particles
cz = 25;
for ix=1:psys.nx
figure;
plot(X{k}(ix,:),'k','LineWidth',2); hold on;
for it=1:t-1
    scatter([it*ones(1,psys.np)],psys.x_1t{ix}(it,:),cz,cols{it},'filled'); hold on;
%     scatter(it,psys.x_1t{ix}(it,:)*psys.weights(:,it),50,'r');hold on;
end
legend('True state','Particles','Weighted estimate')
end
%%
figure;
plot(Mode_model{k},'b','LineWidth',2); hold on;
plot(mode_probable{k},'r--','Linewidth',2);
legend('Modelled mode','Estimated mode')
%%
for im=1:psys.nm
    figure;   
    for it=1:t-1
        scatter([it*ones(1,psys.np)],psys.mu_tt{1}(it,:),25,cols{it},'filled'); hold on;
    end
    plot(mu_weighted{k}(1,:),'r','Linewidth',2); hold on;
    title(['$\mu(M^',num2str(im),')$']);
    legend('Weighter estimate','Particles')
end
