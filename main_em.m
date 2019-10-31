my_init;
%% Define field model parameters
fprintf(1,'Setting model parameters... \n')

% True field (with actual theta)
%basis_type = 'gaussian';
basis_type = 'bspline';
% Set up limits of the grid: x_min,y_min,x_max,y_max
grid_limits = [0, 0, 1000, 1000];

% Set up number of basis functions
nx = 5; ny = 5;
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
%%
grid_limits1 = grid_limits;
Theta = initiate_field(0,60,grid_limits1,knots); %fit splines to a surface
Theta_old = Theta;

 Theta_model = ones(ll,1);
 Theta = ones(ll,1);
 Theta_model(1:4,1) = 10;
 Theta_model(5:8,1) = 100;
 Theta_model(9:12,1) = 190;
 Theta_model(13:16,1) = 290;
 
figure; 
colormap(my_map);
done = plot_field(Theta_model,Z,knots,grid_limits,'bspline');
mu_field = 1;

%%
T = 1; % sampling time
x_len = 4; % size of the state vector
% Transition matrix
I =   eye(2,2);
O = zeros(2,2);

thta = 0.3; % reversion to mean in O-U process 
mean_vel = 0; % mu of O-U process

% brownian motoion with friction  (velocity as O-U process)
F_rw = [I T*I;...
    O   I - thta*T*I]; 
% For CV
F_cv = [I T*I;...
    O   I - thta*T*I];  
 
% Measurement matrix
C = [I O];

% Contrl matrix
B_cv = [O; T*I];
%  B_cv = [O; I];

B_rw = [O; O];

% Disturbance matrix matrix
G_cv = [O; T*I];
% G_rw = [(T*T/2)*I; T*I];
G_rw = [O; T*I];

%% Disturbance matrices

sig1_Q = 2; % RW  disturbance - cell speed
sig2_Q = 2; % CV disturbance - random component of cell acceleration

mm = sig1_Q*G_cv*G_cv';  
dd = diag(mm);
Q_rw = diag(dd);
Q_rw_inv = (1/sig1_Q)*eye(2);
% For DRIFT
mm = sig2_Q*G_cv*G_cv';
dd = diag(mm);
Q_cv = diag(dd);
Q_cv_inv = (1/sig2_Q)*eye(2);
% R - measurement noise
sig2_R = 1;
R = sig2_R*eye(2);
%% Set up Markov chains

n_models = 2;
p_tr = [0.9 0.1;...
        0.1 0.9];...       
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
% for the maximisation step
Sig_w{1} = G{1}*(inv(G{1}'*G{1}))'*Q_cv_inv*inv(G{1}'*G{1})*G{1}';
Sig_w{2} = G{2}*(inv(G{2}'*G{2}))'*Q_rw_inv*inv(G{2}'*G{2})*G{2}';

mu_0 = ones(1,n_models);
mu_0 = mu_0./n_models;

switch_from = zeros(2,1);
switches = zeros(2,2);
nSwitch = 0;

models = [1:100];
for iModel = models
clear X Y Mode_model
load(['simulated_tracks_single_start_' num2str(iModel)]);
Tracks = [1:100]; 
% Tracks = 1; % for now
converged = false;
iter_max  = 10;
iter      = 0;
Theta = zeros(ll,1); 
for k=Tracks
    Z_temp{k} = zeros(4,length(Y{k}));
    Z_temp{k}(1:2,:) = Y{k};
end
%% RUN EM ALGORITHM: MAIN CICLE
while   (iter < iter_max) && ~converged
%% Expectation step
fprintf('E-step... \n')
clear x_merged p_merged mu_s mu x_m x_s P_m P_s
iter = iter + 1;  

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
       clear beta;
       beta = field_gradient(Z_temp{k}(1:2,t-1),Z,knots,basis_type);
       u{t-1,j} = mu_field*beta*Theta;
       [x_m{t,j},P{t,j},lik{t,j}] = kf(y{t},x_0{j},u{t-1,j},P_0{j},F{j},B{j},C,Q{j},R);
       % Nonlinear
%         [x_m{t,j},P{t,j},lik{t,j}] = ekf(y{t},x_0{j},P_0{j},Q{j},R,F{j},B{j},C,Theta,Z,knots,basis_type);
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
   X_filt{k}(:,t) = x_merged{t};    % merged state
   Mu_filt{k}(:,t)  = mu{t}'; 
end
   X_out{k}(:,t) = x_merged{T};    % merged state
   P_out{k,T}(:,:) = P_merged{T};  % merged covariance
   P_for_resampling{k}(T,:,:) = P_merged{T};  % merged covariance
   for j =1:n_models
        X_cond{k,j}(:,T) = x_m{T,j};     % mode-matched posterior estimate
        P_cond{k,j,T}(:,:) = P{T,j};   % mode-matched posterior covariance
   end
%% Recurtion cycle for IMM RTS smoother
% Initialise smoother
mu_s{T} = mu{T};
for j=1:n_models
    x_s{T,j} = x_m{T,j};
    P_s{T,j} = P{T,j};
end
   Mu_out{k}(:,T)  = mu_s{T}';     % mode probability

for t=T-1:-1:1 
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
       beta = field_gradient(Z_temp{k}(1:2,t+1),Z,knots,basis_type);
       u{t,j} = mu_field*beta*Theta;
       [x_s{t,j},P_s{t,j},lik_s{t,j}] =       rts(xs_0{j},x_m{t,j},u{t,j},Ps_0{j},P{t,j},F{j},B{j},Q{j});
%        [x_s{t,j},P_s{t,j}] = erts(xs_0{j},x_m{t,j},Ps_0{j},P{t,j},F{j},B{j},Q{j},Theta,Z,knots,basis_type); % ,lik_s{t,j}
%          [x_s{t,j},P_s{t,j},lik_smm] = urts(xs_0{j},x_m{t,j},Ps_0{j},P{t,j},F{j},B{j},G{j},Q{j},Theta,Z,knots,basis_type);
       
       % Priot to probabilities update
       x_sm_cond_plot{j}(:,t) = x_s{t,j};
       % output to be used in the maximisation
       X_cond{k,j}(:,t) = x_s{t,j};     % mode-matched posterior estimate
       P_cond{k,j,t}(:,:) = P_s{t,j};   % mode-matched posterior covariance
   end
   % Compute smoothed likelihoods
   for j=1:n_models
       lik_s{t,j} = 0;
       for i=1:n_models
           n_s{t+1,i} = sm_lik(x_s{t+1,i},x_m{t,j},P{t,j},F{j},B{j},Q{j},Theta,Z,knots,basis_type);
           lik_s{t,j} = lik_s{t,j} + n_s{t+1,i}*p_tr(i,j);
           lik_s_plot(j,t) = lik_s{t,j};
       end
       m_s(1,j) = lik_s{t,j}*mu{t}(1,j);
   end
   % Mode probabilites 
   d_all = sum(m_s);
   mu_s{t} = m_s./d_all;
   % Merging states
   v = zeros(x_len,1);
   for j=1:n_models
       v = v + mu_s{t}(1,j)*x_s{t,j};
   end
   x_merged{t} = v;
   % Merging covariances
   pv = zeros(x_len);
   for j=1:n_models
       pv = pv + mu_s{t}(1,j)*(P{t,j}+ (x_s{t,j} - v)*(x_s{t,j} - v)');
   end
   P_merged{t} = pv;
   x_sm_plot(:,t) = v;
   mu_plot(:,t) = mu_s{t}';
   [max_mode,i] = max(mu_s{t});
   mode_probable{k}(t) = i;
   
   % outputs of the IMM algorithm
   Lik_out{k}(:,t) = lik_s_plot(:,t);
   X_out{k}(:,t) = x_merged{t};    % merged state
   P_out{k,t}(:,:) = P_merged{t};  % merged covariance
   P_for_resampling{k}(t,:,:) = P_merged{t};  % merged covariance
   Mu_out{k}(:,t)  = mu_s{t}';     % mode probability
end % for smoother (t)
Z_temp{k} = X_out{k};
end % for track (k)

%% Maximization step [2]
fprintf('M-step... \n')
sum1 = 0;
sum2 = 0;
sum3 = 0;
for k=Tracks
      clear dx x1 x2
      [nn,T] = size(X_out{k}); 
      x = X_out{k};
      % CV mode
      x1(:,:) =  X_cond{k,1};
      dx{1}(:,1:T-1) = x1(:,2:T) - F_cv*x(:,1:T-1);
      dx{1}(:,T) = dx{1}(:,T-1);
      % RW mode
      x2(:,:) =  X_cond{k,2};
      dx{2}(:,1:T-1) = x2(:,2:T) - F_rw*x(:,1:T-1);
      dx{2}(:,T) = dx{2}(:,T-1);
     for t=1:T-1
         gg(:,:) = field_gradient(Z_temp{k}(1:2,t),Z,knots,basis_type);
         for j=1:n_models
         bb = B{j}*mu_field*gg;
         sum1 = sum1 + Mu_out{k}(j,t)*bb'*Sig_w{j}*bb;
         sum2 = sum2 + Mu_out{k}(j,t)*bb'*Sig_w{j}*dx{j}(:,t);
         sum3 = sum3 + Mu_out{k}(j,t)*bb'*Sig_w{j}*bb;
         end % for modes (j)
     end % for times (t)
end % for tracks (k)

% Parameter calculation
Theta_old = Theta;
Theta = pinv(sum1)*sum2;
% Expected Fisher information
Expected_info = sum3;
Missing_info = (sum1*Theta - sum2)*(sum1*Theta_old - sum2)';
% temp(:,iter) = Theta;
%%
fprintf('Checking convergence condition... \n')
[converged,dTheta] = converge_param(Theta,Theta_old,iter);
DT_plot(iter) = dTheta;
if converged
    break;
end
end % for EM iterations (iter)
fprintf([num2str(iModel) ' done... \n'])

% Count mode transistions
for k=1:Tracks
    [nn,T] = size(mode_probable{k});  
 for t = 2:T
        switch_from(mode_probable{k}(t-1)) = switch_from(mode_probable{k}(t-1)) + 1;
        switches(mode_probable{k}(t-1),mode_probable{k}(t)) = switches(mode_probable{k}(t-1),mode_probable{k}(t)) + 1;
 end
end
nSwitch = nSwitch +  sum(switch_from);
% Save model paremeters
Theta_all(:,iModel) = Theta;
dTheta_all{iModel} = DT_plot;
Mode_all{iModel} = mode_probable;
Model_model_all{iModel} = Mode_model;
Fisher_info{iModel} = Expected_info;
Miss_fisher{iModel} = Missing_info;
end % for model (iModel)

%%
figure; 
subplot(3,1,1);
plot(x_plot(1,2:end),x_plot(2,2:end)); hold on;
plot(x_sm_plot(1,:),x_sm_plot(2,:)); hold on;
plot(X_cond{100,1}(1,:),X_cond{100,1}(2,:)); hold on;
plot(X_cond{100,2}(1,:),X_cond{100,2}(2,:)); hold on;
subplot(3,1,2);
plot(x_plot(3,:)); hold on;
plot(x_sm_plot(3,:)); hold on;
plot(X_cond{100,1}(3,:)); hold on;
plot(X_cond{100,2}(3,:)); hold on;
subplot(3,1,3); 
plot(x_plot(4,:)); hold on;
plot(x_sm_plot(4,:)); hold on;
plot(X_cond{100,1}(4,:)); hold on;
plot(X_cond{100,2}(4,:)); hold on;
set(findall(gcf,'-property','FontSize'),'FontSize',18);
ChangeInterpreter(gcf,'Latex');


%%
k = 1
figure;
plot(Mode_model{k}); hold on;
plot(mode_probable{k});
legend('Modelled mode','Estimated mode')

figure;
plot(mu_plot(1,:)); hold on;
plot(mu_plot(2,:));
legend('Mode 1','Mode 2')

figure;
plot(lik_plot(1,2:end)); hold on;
plot(lik_s_plot(1,2:end));
legend('Filtered','Smoothed')
title('$\Lambda_1$')

figure;
plot(lik_plot(2,2:end)); hold on;
plot(lik_s_plot(2,2:end));
legend('Filtered','Smoothed')
title('$\Lambda_2$')

%% Compare mode conditioned estimates
figure;
for k = 11
    plot(X{k}(1,:),X{k}(2,:),'k'); hold on;
    plot(X_cond{k,1}(1,:),X_cond{k,1}(2,:),'b'); hold on;
    plot(X_cond{k,2}(1,:),X_cond{k,2}(2,:),'c'); hold on;
    plot(X_out{k}(1,:),X_out{k}(2,:),'r'); hold on;
end
mu_field = 1;

%% Monte-Carlo Results

switches_percent(1:2) = switches(1,:)/switch_from(1);
switches_percent(3:4) = switches(2,:)/switch_from(2);
% cat = categorical({'1 to 1','1 to 2','2 to 1','2 to 2'});
figure;
bar(switches_percent,0.5,'FaceColor',[.9 .9 .9]);
set(gca,'XTickLabel',{'1 to 1','1 to 2','2 to 1','2 to 2'})
ChangeInterpreter(gcf,'Latex');

 %% Mean estimate
for i=1:length(Theta)
    Theta_mean(i,1) = mean(Theta_all(i,:));
    Std_dev(i,1) = std(Theta_all(i,:));
end

%% Variance ellipsoids (now 3D)
FIM100 = zeros(length(Theta),length(Theta));
for iModel=models
    FIM100 = FIM100 + Fisher_info{iModel};
end
COV100 = inv(FIM100);
VOL100 = ellipsoid_volume(ll,0.05,COV100);
TOTALV100 = trace(COV100)/length(Theta);
%
FIM50 = zeros(length(Theta),length(Theta));
for iModel=1:50
    FIM50 = FIM50 + Fisher_info{iModel};
end
COV50 = inv(FIM50);
VOL50 = ellipsoid_volume(ll,0.05,COV50);
for i=1:length(Theta)
    Theta_mean_50(i,1) = mean(Theta_all(i,1:50));
end
TOTALV50 = trace(COV50)/length(Theta);
%
FIM25 = zeros(length(Theta),length(Theta));
for iModel=1:25
    FIM25 = FIM25 + Fisher_info{iModel};
end
COV25 = inv(FIM25);
VOL25 = ellipsoid_volume(ll,0.05,COV25);
for i=1:length(Theta)
    Theta_mean_25(i,1) = mean(Theta_all(i,1:25));
end
TOTALV25 = trace(COV25)/length(Theta);
%
FIM10 = zeros(length(Theta),length(Theta));
for iModel=1:10
    FIM10 = FIM10 + Fisher_info{iModel};
end
COV10 = inv(FIM10);
VOL10 = ellipsoid_volume(ll,0.05,COV10);
for i=1:length(Theta)
    Theta_mean_10(i,1) = mean(Theta_all(i,1:10));
end
TOTALV10 = trace(COV10)/length(Theta);
%
iModel = 1
FIM1 = Fisher_info{iModel};
COV1 = inv(FIM1);
VOL1 = ellipsoid_volume(ll,0.05,COV1);
for i=1:length(Theta)
    Theta_mean_1(i,1) = mean(Theta_all(i,iModel));
end
TOTALV1 = trace(COV1)/length(Theta);
%%
i1 = 1;
i2 = 2;
i3 = 3;

    Theta_hor = [Theta_all(i1,:);Theta_all(i2,:);Theta_all(i3,:)];
    Theta_hor = Theta_hor';
    covar(:,1) = COV100([i1,i2,i3],i1);
    covar(:,2) = COV100([i1,i2,i3],i2);
    covar(:,3) = COV100([i1,i2,i3],i3);
    [X_el,Y_el,Z_el] = build_ellipsoid(Theta_hor,covar);
    [X_el1,Y_el1,Z_el1] = build_ellipsoid_from_data(Theta_hor);
    figure;
    plot3(Theta_hor(:,1),Theta_hor(:,2),Theta_hor(:,3),'o','Color',[80/255, 80/255, 80/255]); hold on;
    plot3(Theta_mean(i1),Theta_mean(i2),Theta_mean(i3),'*r','LineWidth',5); hold on;
    plot3(Theta_model(i1),Theta_model(i2),Theta_model(i3),'*k','LineWidth',5); hold on;
    h = surf(X_el,Y_el,Z_el); alpha 0.8   
    set(h, 'facecolor',[243/255, 222/255, 187/255]);
    set(h, 'edgecolor',[237/255, 177/255, 32/255]);
    hold on;
    h1 = surf(X_el1,Y_el1,Z_el1); alpha 0.5    
    set(h1, 'facecolor',[187/255, 222/255, 243/255]);
    set(h1, 'edgecolor',[32/255, 177/255, 237/255]);
    grid on;
    view(45, 25);
%     xlim([-100,150]); ylim([-100,150]); zlim([-100,150]);
    xlabel(['$\theta_{' num2str(i1) '}$']);
    ylabel(['$\theta_{' num2str(i2) '}$']);
    zlabel(['$\theta_{' num2str(i3) '}$']);
    legend('Estimates','Mean estimate','True value','$95\%$ confidence region','$95\%$ covariance ellipsoid')
    legend('Location','northeast')
	cleanfigure;
% matlab2tikz('ellipses_100_the123.tikz', 'showInfo', false,'parseStrings',false, ...
%          'standalone', false,'height', '3cm', 'width','4cm');    
%%    
Theta_hor = [Theta_all(i1,1:50);Theta_all(i2,1:50);Theta_all(i3,1:50)];
    Theta_hor = Theta_hor';
    covar(:,1) = COV50([i1,i2,i3],i1);
    covar(:,2) = COV50([i1,i2,i3],i2);
    covar(:,3) = COV50([i1,i2,i3],i3);
    [X_el,Y_el,Z_el] = build_ellipsoid(Theta_hor,covar);
    [X_el1,Y_el1,Z_el1] = build_ellipsoid_from_data(Theta_hor);
    figure;
    plot3(Theta_hor(:,1),Theta_hor(:,2),Theta_hor(:,3),'o','Color',[80/255, 80/255, 80/255]); hold on;
    plot3(Theta_mean_50(i1),Theta_mean_50(i2),Theta_mean_50(i3),'*r','LineWidth',5); hold on;
    plot3(Theta_model(i1),Theta_model(i2),Theta_model(i3),'*k','LineWidth',5); hold on;
    h = surf(X_el,Y_el,Z_el); alpha 0.6    
    set(h, 'facecolor',[243/255, 222/255, 187/255]);
    set(h, 'edgecolor',[237/255, 177/255, 32/255]);
    hold on;
    h1 = surf(X_el1,Y_el1,Z_el1); alpha 0.3    
    set(h1, 'facecolor',[187/255, 222/255, 243/255]);
    set(h1, 'edgecolor',[32/255, 177/255, 237/255]);
    grid on;
    view(45, 25);
%     xlim([-100,150]); ylim([-100,150]); zlim([-100,150]);
    xlabel(['$\theta_{' num2str(i1) '}$']);
    ylabel(['$\theta_{' num2str(i2) '}$']);
    zlabel(['$\theta_{' num2str(i3) '}$']);
    legend('Estimates','Mean estimate','True value','$95\%$ confidence region','$95\%$ covariance ellipsoid')
    legend('Location','northeast')
cleanfigure;
matlab2tikz('ellipses_50_the123.tikz', 'showInfo', false,'parseStrings',false, ...
         'standalone', false,'height', '3cm', 'width','4cm');

%%    
Theta_hor = [Theta_all(i1,1:25);Theta_all(i2,1:25);Theta_all(i3,1:25)];
    Theta_hor = Theta_hor';
    covar(:,1) = COV25([i1,i2,i3],i1);
    covar(:,2) = COV25([i1,i2,i3],i2);
    covar(:,3) = COV25([i1,i2,i3],i3);
    [X_el,Y_el,Z_el] = build_ellipsoid(Theta_hor,covar);
    [X_el1,Y_el1,Z_el1] = build_ellipsoid_from_data(Theta_hor);
    figure;
    plot3(Theta_hor(:,1),Theta_hor(:,2),Theta_hor(:,3),'o','Color',[80/255, 80/255, 80/255]); hold on;
    plot3(Theta_mean_25(i1),Theta_mean_25(i2),Theta_mean_25(i3),'*r','LineWidth',5); hold on;
    plot3(Theta_model(i1),Theta_model(i2),Theta_model(i3),'*k','LineWidth',5); hold on;
    h = surf(X_el,Y_el,Z_el); alpha 0.8    
    set(h, 'facecolor',[243/255, 222/255, 187/255]);
    set(h, 'edgecolor',[237/255, 177/255, 32/255]);
    hold on;
    h1 = surf(X_el1,Y_el1,Z_el1); alpha 0.5    
    set(h1, 'facecolor',[187/255, 222/255, 243/255]);
    set(h1, 'edgecolor',[32/255, 177/255, 237/255]);
    grid on;
    view(45, 25);
%     xlim([-100,150]); ylim([-100,150]); zlim([-100,150]);
    xlabel(['$\theta_{' num2str(i1) '}$']);
    ylabel(['$\theta_{' num2str(i2) '}$']);
    zlabel(['$\theta_{' num2str(i3) '}$']);
    legend('Estimates','Mean estimate','True value','$95\%$ confidence region','$95\%$ covariance ellipsoid')
    legend('Location','northeast')
cleanfigure;
matlab2tikz('ellipses_25_the123.tikz', 'showInfo', false,'parseStrings',false, ...
         'standalone', false,'height', '3cm', 'width','4cm');
%%    
Theta_hor = [Theta_all(i1,1:10);Theta_all(i2,1:10);Theta_all(i3,1:10)];
    Theta_hor = Theta_hor';
    covar(:,1) = COV10([i1,i2,i3],i1);
    covar(:,2) = COV10([i1,i2,i3],i2);
    covar(:,3) = COV10([i1,i2,i3],i3);
    [X_el,Y_el,Z_el] = build_ellipsoid(Theta_hor,covar);
    [X_el1,Y_el1,Z_el1] = build_ellipsoid_from_data(Theta_hor);
    figure;
    plot3(Theta_hor(:,1),Theta_hor(:,2),Theta_hor(:,3),'o','Color',[80/255, 80/255, 80/255]); hold on;
    plot3(Theta_mean_10(i1),Theta_mean_10(i2),Theta_mean_10(i3),'*r','LineWidth',5); hold on;
    plot3(Theta_model(i1),Theta_model(i2),Theta_model(i3),'*k','LineWidth',5); hold on;
    h = surf(X_el,Y_el,Z_el); alpha 0.8    
    set(h, 'facecolor',[243/255, 222/255, 187/255]);
    set(h, 'edgecolor',[237/255, 177/255, 32/255]);
    hold on;
    h1 = surf(X_el1,Y_el1,Z_el1); alpha 0.5    
    set(h1, 'facecolor',[187/255, 222/255, 243/255]);
    set(h1, 'edgecolor',[32/255, 177/255, 237/255]);
    grid on;
    view(45, 25);
%     xlim([-100,150]); ylim([-100,150]); zlim([-100,150]);
    xlabel(['$\theta_{' num2str(i1) '}$']);
    ylabel(['$\theta_{' num2str(i2) '}$']);
    zlabel(['$\theta_{' num2str(i3) '}$']);
    legend('Estimates','Mean estimate','True value','$95\%$ confidence region','$95\%$ covariance ellipsoid')
    legend('Location','northeast')
    cleanfigure;
matlab2tikz('ellipses_10_the123.tikz', 'showInfo', false,'parseStrings',false, ...
         'standalone', false,'height', '3cm', 'width','4cm');
%%
for iPair=1:2:ll-1
    cc(:,1) = COV100([iPair, iPair+1],iPair);
    cc(:,2) = COV100([iPair, iPair+1],iPair+1);
    co{iPair} = cc;
    mea{iPair} = [mean(Theta_all(iPair,:)),mean(Theta_all(iPair+1,:))];
    ell_points_theta{iPair} = build_ellips(mea{iPair},co{iPair});
end

for iPair=1:2:ll-1
    figure;
    plot(Theta_all(iPair,:),Theta_all(iPair+1,:),'*k'); hold on;
    plot(Theta_model(iPair),Theta_model(iPair+1),'*b','Linewidth',3); hold on;
    plot(mea{iPair}(1),mea{iPair}(2),'*r','Linewidth',3); hold on;
    plot(ell_points_theta{iPair}(:,1),ell_points_theta{iPair}(:,2),'r');
    xlabel(['$\Theta_{' num2str(iPair) '}$']);
    ylabel(['$\Theta_{' num2str(iPair+1) '}$']);
end
%%
% fig - modeled field
figure; 
colormap(my_map);
done1 = plot_heatmap(Theta_model,Z,knots,grid_limits,basis_type);
hold on;
side = knots(1,2) - knots(1,1);
for i=1:length(Theta_model)
    a = knots(1,i*2-1);
    b = knots(2,i*2-1);
    p=rectangle('Position',[a,b,side,side],'Curvature',0.1,'EdgeColor','w'); 
    hold on;
end
ChangeInterpreter(gcf,'Latex');


%% 
for iModel=models
    Theta_temp = Theta_all(:,iModel);
    Dist(iModel) = norm(Theta_model - Theta_temp);
end
[D_min,I_min] = min(Dist);
Theta_closest =  Theta_all(:,I_min);
[D_min,I_max] = max(Dist);
Theta_furthest =  Theta_all(:,I_max);
Theta_median = median(Theta_all');
%%
% Plot b-spline grid in 3d
x0 = 20;
y0 = 20;
width=700;
height=500;
figure; 
colormap(my_map);
set(gcf,'units','points','position',[x0,y0,width,height]);
z_ticklabels{1} = '';
k = 1;
side1 = knots(1,2) - knots(1,1);
side2 = knots(2,2) - knots(2,1);
z_ticklabels{k+1} = num2str(k-1);
for j=1:length(Theta_model)
    a = knots(1,j*2-1);
    b = knots(2,j*2-1);
    Theta_temp = zeros(length(Theta_model),1);
    Theta_temp(j) = Theta_model(j);
    k = k + 1;
    z_ticklabels{k+1} = num2str(k-1);
    gridlim = [a b a+side1 b+side2];
    done = plot_surface(Theta_temp,Z,knots,grid_limits,basis_type);
    hold on;
    txt = (['$\theta_{' num2str(j) '} =$' num2str(Theta_temp(j))]);
    xx = a + side1/2 - 30;
    yy = b + side2/2 - 30;
    hh = Theta_temp(j);
    text(xx,yy,hh+5,txt,'Color','k','FontSize',18)
end
% view(3)
az = -30;
el = 40;
view(az, el);
zlim([0 300]);
z_ticks = [1:1:k];
set(gca,'ZTick',[])
set(gca,'XTick',[])
set(gca,'YTick',[])
set(gca,'ZTickLabel',z_ticklabels)
zlabel('Grid of basis functions')
ChangeInterpreter(gcf,'Latex');
set(findall(gcf,'-property','FontSize'),'FontSize',24)
%%
Theta_diff = Theta_model - Theta_mean;
% fig - difference
figure; 
% colormap(my_map);
[done1,ZZ] = plot_heatmap(Theta_diff,Z,knots,grid_limits,basis_type);
hold on;
side = knots(1,2) - knots(1,1);
for i=1:length(Theta_mean)
    a = knots(1,i*2-1);
    b = knots(2,i*2-1);
    p=rectangle('Position',[a,b,side,side],'Curvature',0.1,'EdgeColor','w'); 
    hold on;
end
xlabel('\textrm{X, a.u.}', 'interpreter', 'latex');
ylabel('\textrm{Y,  a.u.}', 'interpreter', 'latex');
set(findall(gcf,'-property','FontSize'),'FontSize',18);
ChangeInterpreter(gcf,'Latex');
%%
% fig - estimated field (mean)
figure; 
colormap(my_map);
[done2,ZZ] = plot_heatmap(Theta_model,Z,knots,grid_limits,basis_type);
hold on;
side = knots(1,2) - knots(1,1);
for i=1:length(Theta)
    a = knots(1,i*2-1);
    b = knots(2,i*2-1);
    p=rectangle('Position',[a,b,side,side],'Curvature',0.1,'EdgeColor','w'); 
    hold on;
end
xlabel('\textrm{X, a.u.}', 'interpreter', 'latex');
ylabel('\textrm{Y,  a.u.}', 'interpreter', 'latex');
set(findall(gcf,'-property','FontSize'),'FontSize',18);
ChangeInterpreter(gcf,'Latex');

%%
figure; 
colormap(my_map);
[done3, ZZ1] = plot_gradient(Theta_model,Z,knots,grid_limits,basis_type);
hold on;
% line([450, 450],[450, 450],'Color','m','LineWidth',4);
% line([450, 450],[450, 450],'Color','m','LineWidth',4);
colorbar
xlim([0 1000]);
ylim([0 1000]);
% caxis([0 500]);
xlabel('\textrm{X,  a.u.}', 'interpreter', 'latex');
ylabel('\textrm{Y,  a.u.}', 'interpreter', 'latex');
set(findall(gcf,'-property','FontSize'),'FontSize',18);
ChangeInterpreter(gcf,'Latex');

%%
figure; 
% colormap(my_map);
[done4,ZZ] = plot_gradient(Theta_mean,Z,knots,grid_limits,basis_type);
hold on;
 line([450, 450],[250, 750],'Color','m','LineWidth',4);
%  scatter(450,450,50,'m','filled');
colormap jet
colorbar
xlim([0 1000]);
ylim([0 1000]);
%  caxis([0 500]);
xlabel('\textrm{X, a.u.}', 'interpreter', 'latex');
ylabel('\textrm{Y,  a.u.}', 'interpreter', 'latex');
set(findall(gcf,'-property','FontSize'),'FontSize',18);
ChangeInterpreter(gcf,'Latex');

%%
figure; 
colormap(my_map);
done4 = plot_gradient(Theta,Z,knots,grid_limits,basis_type);
hold on;
for k=1:20
    plot(X{k}(1,:),X{k}(2,:),'k'); hold on;
    c = Mu_filt{k}(1,2:end);
    cplot(X_filt{k}(1,2:end),X_filt{k}(2,2:end),c,'-',...
    'markerfacecolor','flat',...
    'markeredgecolor','w','linewidth',2.0); hold on;
end
colorbar;
xlim([230 700]);
ylim([300 800]);
xlabel('\textrm{X,  a.u.}', 'interpreter', 'latex');
ylabel('\textrm{Y,  a.u.}', 'interpreter', 'latex');
caxis([0 1]);
set(findall(gcf,'-property','FontSize'),'FontSize',18);
ChangeInterpreter(gcf,'Latex');
%%
figure; 
colormap(my_map);
done5 = plot_gradient(Theta,Z,knots,grid_limits,basis_type);
hold on;
for k=1:20
    plot(X{k}(1,:),X{k}(2,:),'k'); hold on;
    c = Mu_out{k}(1,:);
    cplot(X_out{k}(1,:),X_out{k}(2,:),c,'-',...
    'markerfacecolor','flat',...
    'markeredgecolor','w','linewidth',2.0); hold on;
end
colorbar;
caxis([0 1]);
xlim([230 700]);
ylim([300 800]);
xlabel('\textrm{X,  a.u.}', 'interpreter', 'latex');
ylabel('\textrm{Y,  a.u.}', 'interpreter', 'latex');
set(findall(gcf,'-property','FontSize'),'FontSize',18);
ChangeInterpreter(gcf,'Latex');

