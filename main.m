my_init
%% Define field model parameters
fprintf(1,'Setting model parameters... \n')

% True field (with actual theta)
%basis_type = 'gaussian';
basis_type = 'bspline';
% Set up limits of the grid: x_min,y_min,x_max,y_max
grid_limits = [0, 0, 1000, 1000];

% Set up number of basis functions
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
Theta = initiate_field(0,60,grid_limits1,knots); %fit splines to a surface
Theta_old = Theta;

 Theta = ones(ll,1);
 Theta(1:4,1) = 10;
 Theta(5:8,1) = 20;
 Theta(9:12,1) = 30;
 Theta(13:16,1) = 40;
 
done = plot_field(Theta,Z,knots,grid_limits,'bspline');
mu_field = 1;


dT = 1; % sampling time
x_len = 4; % size of the state vector
% Transition matrix
I =   eye(2,2);
O = zeros(2,2);

thta = 0.1; % reversion to mean in O-U process 
thta1 = 0.6; % reversion to mean in O-U process 

mean_vel = 0; % mu of O-U process

% Brownian motoion with friction  (velocity as O-U process)
F_rw = [I            dT*I;...
        O   I - thta1*dT*I]; 
% For CV
F_cv = [I dT*I;...
        O   I  - thta*dT*I];  %
% Observation matrix
C = [I O];
% Contrl matrix
B_cv = [dT^2/2*I; dT*I];
%  B_cv = [O; dT*I];
B_rw = [O; O];

% Disturbance matrix matrix
G_cv = [dT^2/2*I; dT*I];
% G_rw = [(T*T/2)*I; T*I];
G_rw = [dT^2/2*I; dT*I];

W = [I; O];

%% Disturbance matrices

sig1_Q = 2; % RW  disturbance - cell speed
sig2_Q = 1; % CV disturbance - random component of cell acceleration

mm = sig1_Q*G_cv*G_cv';  
dd = diag(mm);
Q_rw = diag(dd);
% For DRIFT
mm = sig2_Q*G_cv*G_cv';
dd = diag(mm);
Q_cv = diag(dd);
% R - measurement noise
sig2_R = 2;
R = sig2_R*eye(2);
%% Set up Markov chains

n_models = 2;
p_tr = [0.95 0.1;...
        0.05 0.9];...       
% P_tr = [1 0;...
%         0 1];...   

F{1} = F_cv;
F{2} = F_rw;
B{1} = B_cv;
B{2} = B_rw;
Q{1} = Q_cv;
Q{2} = Q_rw;
G{1} = G_cv;
G{2} = G_rw;

mu_0 = ones(1,n_models);
mu_0 = mu_0./n_models;


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


%% recurtion cycle for IMM filter
mu{1} = mu_0;
for j=1:n_models
    x_m{1,j} = zeros(x_len,1);
    x_m{1,j}(1:2,:) = y{1};
    P{1,j} = eye(x_len);
end

for t=2:T
    % Mode conditioned calculations
   for j=1:n_models
       % Calculate mixing probabilities
       c(j) = mu{t-1}*p_tr(:,j); % normalising constant
       for i=1:n_models 
           mu_mix(i,j) = p_tr(i,j)*mu{t-1}(i)/c(j); % probabilities
       end
       % Mixing initial conditions for filtering
       v = zeros(x_len,1);
       for i=1:n_models
           v = v +  x_m{t-1,j}*mu_mix(i,j);
       end
       pv = zeros(x_len);
       for i=1:n_models
           pv = pv +  mu_mix(i,j)*(P{t-1,j} + (x_m{t-1,j} - v)*(x_m{t-1,j} - v)');
       end
       x_0{j} = v;
       P_0{j} = pv;
       % Model-matched filtering
       % Linear
%        clear beta;
%        beta = field_gradient(x_0{j}(1:2),Z,knots,basis_type);
%        u{t-1,j} = mu_field*beta*Theta;
%        [x_m{t,j},P{t,j},lik{t,j}] = kf(y{t},x_0{j},u{t-1,j},P_0{j},F{j},B{j},C,Q{j},R);
       % Nonlinear
         [x_m{t,j},P{t,j},lik{t,j}] = ekf(y{t},x_0{j},P_0{j},Q{j},R,F{j},B{j},C,Theta,Z,knots,basis_type);
%         [x_pr,x_m{t,j},P_pr,P{t,j},lik{t,j}] = ukf(y{t},x_0{j},P_0{j},Q{j},R,F{j},B{j},C,Theta,Z,knots,basis_type);            

       % Prior to probabilities update
       m(1,j) = lik{t,j}*c(j);
       lik_plot(j,t) = lik{t,j};
       x_cond_plot{j}(:,t) = x_m{t,j};
   end
   % Mode probabilites 
   c_all = sum(m);
   mu{t} = m./c_all;
   % Merging states (optional)
   v = zeros(x_len,1);
   for j=1:n_models
       v = v + mu{t}(1,j)*x_m{t,j};
   end
   x_merged{t} = v;
   % Merging covariances (optional)
   pv = zeros(x_len);
   for j=1:n_models
       pv = pv + mu{t}(1,j)*(P{t,j}+ (x_m{t,j} - v)*(x_m{t,j} - v)');
   end
   P_merged{t} = pv;
   x_plot(:,t) = v;
   mu_plot(:,t) = mu{t}';
   [max_mode,i] = max(mu{t});
   mode_probable{k}(t) = i;
end

%% Recurtion cycle for IMM RTS smoother
% Initialise smoother
mu_s{T} = mu_0;
for j=1:n_models
    x_s{T,j} = x_m{T,j};
    P_s{T,j} = P{T,j};
end

for t=T-1:-1:1
    % Backward time probabilities
    
    % Mode conditioned calculations
   for j=1:n_models
       % Calculate mixing probabilities
       d(1,j) = mu_s{t+1}*p_tr(:,j); % normalising constant
       for i=1:n_models 
           mu_mix(i,j) = p_tr(i,j)*mu_s{t+1}(i)/d(j); % probabilities
       end
       % Mixing initial conditions for filtering
       v = zeros(x_len,1);
       for i=1:n_models
           v = v +  x_s{t+1,j}*mu_mix(i,j);
       end
       pv = zeros(x_len);
       for i=1:n_models
           pv = pv +  mu_mix(i,j)*(P_s{t+1,j} + (x_s{t+1,j} - v)*(x_s{t+1,j} - v)');
       end
       xs_0{j} = v;
       Ps_0{j} = pv;
       % Model-matched smoothing
       % Linear
%        [x_s{t,j},P_s{t,j},lik_s{t,j}] = rts(xs_0{j},x_m{t,j},u{t,j},Ps_0{j},P{t,j},F{j},B{j},Q{j});
         [x_s{t,j},P_s{t,j},lik_s{t,j}] = erts(xs_0{j},x_m{t,j},Ps_0{j},P{t,j},F{j},B{j},Q{j},Theta,Z,knots,basis_type);
%          [x_s{t,j},P_s{t,j},lik_s{t,j}] = urts(xs_0{j},x_m{t,j},Ps_0{j},P{t,j},F{j},B{j},G{j},Q{j},Theta,Z,knots,basis_type);
       % Priot to probabilities update
       m_s(1,j) = lik_s{t,j}*d(j);
       lik_s_plot(j,t) = lik_s{t,j};
       x_sm_cond_plot{j}(:,t) = x_s{t,j};
   end
   % Mode probabilites 
   d_all = sum(m_s);
   mu_s{t} = m_s./d_all;
   % Merging states (optional)
   v = zeros(x_len,1);
   for j=1:n_models
       v = v + mu{t}(1,j)*x_s{t,j};
   end
   x_merged{t} = v;
   % Merging covariances (optional)
   pv = zeros(x_len);
   for j=1:n_models
       pv = pv + mu{t}(1,j)*(P{t,j}+ (x_s{t,j} - v)*(x_s{t,j} - v)');
   end
   P_merged{t} = pv;
   x_sm_plot(:,t) = v;
   mu_plot(:,t) = mu_s{t}';
   [max_mode,i] = max(mu_s{t});
   mode_probable{k}(t) = i;
end % for smoother (t)
end % for track (k)



end % for model (iModel)

%% 
%% 
figure; 
plot(X{k}(1,:),X{k}(2,:),'k','LineWidth',2); hold on;
xlabel('X, a.u.'); ylabel('Y, a.u.');
plot(x_plot(1,2:end),x_plot(2,2:end)); hold on;
plot(x_sm_plot(1,:),x_sm_plot(2,:)); hold on;
legend('true','filtered','smoothed')

%% 
for ix=1:x_len
figure;
colormap(my_map);
plot(X{k}(ix,:),'k','LineWidth',2); hold on;
plot(x_plot(ix,2:end),'LineWidth',2); hold on;
plot(x_cond_plot{1}(ix,2:end),'LineWidth',2); hold on;
plot(x_cond_plot{2}(ix,2:end),'LineWidth',2); hold on; 
plot(x_sm_plot(ix,:),'LineWidth',2); hold on;
plot(x_sm_cond_plot{1}(ix,2:end),'LineWidth',2); hold on;
plot(x_sm_cond_plot{2}(ix,2:end),'LineWidth',2); hold on; 

legend('real','filtered','mode 1','mode 2','smoothed','smoothed 1','smoothed 2');
end
%%
figure;
plot(Mode_model{k},'b','LineWidth',2); hold on;
plot(mode_probable{k},'r--','Linewidth',2);
legend('Modelled mode','Estimated mode')

figure;
plot(mu_plot(1,:)); hold on;
plot(mu_plot(2,:));
legend('Mode 1','Mode 2')

figure;
plot(lik_plot(1,2:end)); hold on;
plot(lik_s_plot(1,2:end));
legend('Filtered','Smoothed')

figure;
plot(lik_plot(2,2:end)); hold on;
plot(lik_s_plot(2,2:end));
legend('Filtered','Smoothed')