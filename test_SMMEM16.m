% A.R. Kadochnikova - 30/04/2017

% Simultaneous cell mode identification and chemotactic field inferrence.
% State estimation/mode identification carried out through MMF
% Filed inferrence carried out through EM algorithm.

% file "measured_data_" containes cell tracks 

% [1] V. Kadirkamanathan, G.R. Holmes "The Neutrophils Eye-View: Inference
% and Visualisation of the Chemoattractant field Driving Cell Chemotaxis", 2012. 
% [2] V. Kadirkamanathan, S. Anderson "Maximum-Likelihood Estimation of
% Delta-Domain Model Parameters from Noisy Output Signals", 2008.
% [3] Kalman Smoother without inv(P) A.Logothesis and V.Krishnamurthy "EM 
% alforithms for MAP estimation of jump Markov linear systems", 1999.
% [4] Multiple model Kalman filter from X. Zhang "Modelling and
% Identification of Neutrophil Cell Dynamic Behaviour", 2016.

clc; clear;
close all;

nModels = 50;
%% Field model parameters

% Type of basis: gaussian or spline
%basis_type = 'gaussian';
basis_type = 'bspline';

% Set up limits of the grid: x_min,y_min,x_max,y_max
grid_limits = [0, 0, 900, 900];

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
Theta = initiate_field(0,0,grid_limits1,knots); %fit splines to a surface
Theta_old = Theta;
% Theta = zeros(nx*ny,1);
% Propotional coefficient (sensitivity)
mu = 1;

%% State-space model parameters
% Two candidate models of cell dynamics are examined: 
%random walk ('rw') and CV model ('ks')

fprintf(1,'Setting model parameters... \n')
T = 1; % sampling time

% Transition matrix
I =   eye(2,2);
O = zeros(2,2);
% For RW (with I for second order, with O for first order)
F_rw = [I T*I;...
    O   I]; 
% For CV
F_cv = [I T*I;...
    O   I];  
 
% Measurement matrix
H = [I O];

% Contrl matrix
B_cv = [(T*T/2)*I; T*I];
B_rw = [O; O];

% Disturbance matrix matrix
G_cv = [(T*T/2)*I; T*I];
G_rw = [(T*T/2)*I; T*I];

%% Disturbance matrices

% Q - describes power of state noise
sig1_Q = 1; % RW  disturbance - cell speed
sig2_Q = 0.09; % CV disturbance - random component of cell acceleration
% For DIFF
Q_rw = sig1_Q*G_rw*G_rw';  
% d = diag(m);
% Q_rw = diag(d);
Q_rw_inv = (1/sig1_Q)*eye(2);
% For DRIFT
Q_cv = sig2_Q*G_cv*G_cv';
% d = diag(m);
% Q_cv = diag(d);
Q_cv_inv = (1/sig2_Q)*eye(2);
% For STAT
sig2_R = 0.2;
R = eye(2);
% P - prior covariance
P = eye(4);

%% Set up immf
n_models = 2;
% P_tr = [0.9 0.1;...
%         0.1 0.9];...
        
P_tr = [0.9 0.1;...
        0.1 0.9];...
        
    
F{1} = F_cv;
F{2} = F_rw;
B{1} = B_cv;
B{2} = B_rw;
Q{1} = Q_cv;
Q{2} = Q_rw;
G{1} = G_cv;
G{2} = G_rw;

% for backward filtering
Q_w{1} = Q_cv;
Q_w{2} = Q_rw;
Sig_w{1} = G{1}*(inv(G{1}'*G{1}))'*Q_cv_inv*inv(G{1}'*G{1})*G{1}';
Sig_w{2} = G{2}*(inv(G{2}'*G{2}))'*Q_rw_inv*inv(G{2}'*G{2})*G{2}';

%% Mode switch probabilities
nSwitch = 0;
% top row = transition from mode 1; bottom row = transtion from mode 2.
switches = zeros(2,2);
switch_from = zeros(1,2);
Theta_all = zeros(length(Theta),nModels);

for iModel = 1:nModels
clear X Y nTracks
    
    load(['simulated_16_' num2str(iModel)]);

    nTracks = 100;
%% Estimation framework
converged = false;
iter_max  = 10;
iter      = 0;
%% RUN EM ALGORITHM: MAIN CICLE
while   (iter < iter_max) && ~converged

clear X_smoothed X_rts P_rts X_mode X_resample MU_mem X_posterior P_posterior X_resample P_resample
iter = iter + 1;    
%% Expectation step

% State Estimation
% For each datapoint keep in workspace:
% Prior and posterior Kalman estimates
% Prior and posterior covariances
% Field gradient in the point (in meausred coordinates of cell centroids Y)
% Smoothed estimate and final posterior covariance

fprintf('Running IMM framework... \n')
for j = 1:nTracks
    [n,l] = size(X{j});
    %% IMM filter
    % initialise filters
    X_mode = zeros(n,1);
    x_posterior   = [Y{j}(1,:)' ; 0; 0];
    p_posterior   = eye(4,4);
    X_posterior{j}(1,:) = x_posterior;
    P_posterior{j}(1,:,:) = p_posterior;
    MU_mem{j} = zeros(n,n_models);
    Mu(1:n_models) = 1/n_models;
    MU_mem{j}(1,:) = Mu;
%     RMSE{j,iModel}(1,1) = sqrt(mean((X_posterior{j}(1,1) - X{j}(1,1)).^2));
%     RMSE{j,iModel}(2,1) = sqrt(mean((X_posterior{j}(1,2) - X{j}(1,2)).^2));
%     RMSE{j,iModel}(3,1) = sqrt(mean((X_posterior{j}(1,3) - X{j}(1,3)).^2));
%     RMSE{j,iModel}(4,1) = sqrt(mean((X_posterior{j}(1,4) - X{j}(1,4)).^2));
    for k=1:n_models
        x_kk(:,k,1) = x_posterior;
    end
    % filters
    for i = 2:n+1 
        gg(:,:) = field_gradient(x_posterior(1:2),Z,knots,basis_type);
        u = mu*gg*Theta;
        for k=1:n_models
            if i == n+1
                lll = i-1;
                [x_pr(:,k,i),x_kk(:,k,i),p_pr(:,:,k,i),p_kk(:,:,k,i),Lkl(k,i)] = kalman_filter(Y{j}(lll,:),x_posterior,u,p_posterior,F{k},H,Q{k},R,B{k});
            else
                 [x_pr(:,k,i),x_kk(:,k,i),p_pr(:,:,k,i),p_kk(:,:,k,i),Lkl(k,i)] = kalman_filter(Y{j}(i,:),x_posterior,u,p_posterior,F{k},H,Q{k},R,B{k});
            end
           
        end
        % mixer 
        [Mu,x_posterior,p_posterior] = mixer(Lkl(:,i),Mu,P_tr,x_kk(:,:,i),p_kk(:,:,:,i));
        [Mu_m,k_max] = max(Mu);
        % Save to cells
        X_kk{j}(:,:,i-1) = x_kk(:,:,i);
        X_prior    {j}(i-1,:)   = x_pr(:,k_max,i)';
        X_posterior{j}(i-1,:)   = x_posterior';
        P_prior    {j}(i-1,:,:) = p_pr(:,:,k_max,i);
        P_posterior{j}(i-1,:,:) = p_posterior;
        X_mode1{j}(i-1) = mode_identification(Mu);
        MU_mem{j}(i-1,:) = Mu;
        
        RMSE{iter}(j,iModel,1,i-1) = sqrt(mean((X_posterior{j}(i-1,1) - X{j}(i-1,1)).^2));
        RMSE{iter}(j,iModel,2,i-1) = sqrt(mean((X_posterior{j}(i-1,2) - X{j}(i-1,2)).^2));
        RMSE{iter}(j,iModel,3,i-1) = sqrt(mean((X_posterior{j}(i-1,3) - X{j}(i-1,3)).^2));
        RMSE{iter}(j,iModel,4,i-1) = sqrt(mean((X_posterior{j}(i-1,4) - X{j}(i-1,4)).^2));
    end
    
%     %% IMM smoothing
%     % initialise smoother
%     mu_s{j}(n,1:n_models) = Mu;
%     p_kk1 = zeros(4,4,n_models,n);
%     x_kk1 = zeros(4,n_models,n);
%     X_smth{j} = zeros(4,n);
%     P_smth{j} = zeros(n,4,4);
%     X_smth{j}(:,n) = X_posterior{j}(n,:)';
%     P_smth{j}(n,:,:) = P_posterior{j}(n,:,:);
%     X_mode(n) = mode_identification(mu_s{j}(n,:));
%     %init backward estimates
%     for k=1:n_models
%         p_kk1(:,:,k,n) = eye(4,4); %pinv(H'*pinv(R)*H); 
%         x_kk1(:,k,n) =  p_kk1(:,:,k,n)*H'*pinv(R)*Y{j}(n,:)';
%     end
%     for i=n-1:-1:1
%         gg(:,:) = field_gradient(X_posterior{j}(i,:),Z,knots,basis_type);
%         u = mu*gg*Theta;
%         % backward estimates
%     for k=1:n_models
%         x_kk1(:,k,i) = pinv(F{k})*(x_kk1(:,k,i+1) - B{k}*u);
%         p_kk1(:,:,k,i)= pinv(F{k})*(p_kk1(:,:,k,i+1) + G{k}*Q_w{k}*G{k}')*(pinv(F{k}))';
%     end
%     % smoother and mixer
%     [X_smth{j}(:,i),P_smth{j}(i,:,:),Likl{j}(:,:,i),mu_s{j}(i,:)] = immsmoother(x_kk(:,:,i),p_kk(:,:,:,i),x_kk1(:,:,i),p_kk1(:,:,:,i),MU_mem{j}(i,:),P_tr);
%     X_mode(i) = mode_identification(mu_s{j}(i,:));
%     end
%     
%     X_imms{j} = X_smth{j}(:,2:n)';
%     P_imms{j} = P_smth{j}(2:n,:,:);
%     Mode_imms{j} = X_mode(2:n);
    
    X_rts{j} = X_posterior{j};
    P_rts{j} = P_posterior{j};
    Mode{j} = X_mode1{j};
    clear dx

    % save displacements for hystograms 
    [n,l] = size(X_rts{j});  
    dx_plot{j} = zeros(n,l);
    dx(1:n-1,:) = X_rts{j}(2:n,:) - X_rts{j}(1:n-1,:);
    dx(n,:) = dx(n-1,:);
    dx_plot{j} = dx;
end

% resample tracks based on cell mode
% [X_resample,P_resample,nResampled,Grad] = resample_tracks(X_rts,P_rts,Mode,1,ll,nTracks,iter,Z,knots,basis_type);

%% Maximization step [2]

sum1 = 0;
sum2 = 0;
sum3 = 0;

for j=1:nTracks
    clear dx x1 x2
     [n,l] = size(X_rts{j});  
      x = X_rts{j}';
      
      % CV mode
      x1(:,:) =  X_kk{j}(:,1,:);
      dx{1}(:,1:n-1) = x1(:,2:n) - F_cv*x(:,1:n-1);
      dx{1}(:,n) = dx{1}(:,n-1);
      % RW mode
      x2(:,:) =  X_kk{j}(:,2,:);
      dx{2}(:,1:n-1) = x2(:,2:n) - F_rw*x(:,1:n-1);
      dx{2}(:,n) = dx{2}(:,n-1);
     for i=2:n
         for l=1:n_models
         gg(:,:) = field_gradient(X_rts{j}(i,1:2)',Z,knots,basis_type);
         bb = B{l}*mu*gg;
         sum1 = sum1 + MU_mem{j}(i-1,l)*bb'*Sig_w{l}*bb;
         sum2 = sum2 + MU_mem{j}(i,l)*bb'*Sig_w{l}*dx{l}(:,i);
         sum3 = sum3 + MU_mem{j}(i-1,l)*bb'*Sig_w{l}*bb;
         end
     end
end

fprintf('Calculating parameters... \n')
% Updating parameter vector
Theta_old = Theta;
Theta = pinv(sum1)*sum2;
Fisher_info = sum3;
% temp(:,iter) = Theta;

% Check parameter convergence
fprintf('Checking convergence condition... \n')
[converged,dTheta] = converge_param(Theta,Theta_old,iter);
DT_plot(iter) = dTheta;
if converged
    break;
end

end
% Count mode transistions
for j=1:nTracks
    [l,n] = size(X_mode1{j});  
 for i = 2:n
        switch_from(X_mode1{j}(i-1)) = switch_from(X_mode1{j}(i-1)) + 1;
        switches(X_mode1{j}(i-1),X_mode1{j}(i)) = switches(X_mode1{j}(i-1),X_mode1{j}(i)) + 1;
 end
end
nSwitch = nSwitch +  sum(switch_from);
% Save model paremeters
Theta_all(:,iModel) = Theta;
dTheta_all{iModel} = DT_plot;
Mode_all{iModel} = Mode;
Model_model_all{iModel} = Mode_model;
Fisher_all{iModel} = Fisher_info;
end

% Overall histograms (velocities)
Vx_count = 1;
Vy_count = 1;
Sx_count = 1;
Sy_count = 1;
for j = 1:nTracks
   [n,l] = size(X_rts{j});
   if n>l
   Sx_hist(1,Sx_count:Sx_count+n-1) = X_rts{j}(:,1)./T;
   Sy_hist(1,Sy_count:Sy_count+n-1) = X_rts{j}(:,2)./T;
   Dx_hist(1,Sx_count:Sx_count+n-1) = dx_plot{j}(:,1)./T;
   Dy_hist(1,Sy_count:Sy_count+n-1) = dx_plot{j}(:,2)./T;
   Vx(1,Vx_count:Vx_count+n-1) = X_rts{j}(:,3)'./T;
   Vy(1,Vy_count:Vy_count+n-1) = X_rts{j}(:,4)'./T;
   Sx_count = Sx_count + n;
   Sy_count = Sy_count + n;
   Vx_count = Vx_count + n;
   Vy_count = Vy_count + n;
   end
end

%save LL_new_model_final LL_current

%% Plots
greeks = Greeks; % load in all greek symbols. Thanks to: Oleg Komarov

% Velocities
numberOfBins = 50;

figure; set(gcf,'color','w');
% title('Velocity histogram')
[counts, binValues] = hist(Vx, numberOfBins);
normalizedCounts = 100 * counts / sum(counts);
subplot(2,1,1);
bar(binValues, normalizedCounts, 'barwidth', 1,'FaceColor',[.9 .9 .9]);
xlim([-20 20])
xlabel('\textrm{Vx,$\mu$m/min}', 'interpreter', 'latex');
ylabel('\textrm{$\%$}', 'interpreter', 'latex');
[counts, binValues] = hist(Vy, numberOfBins);
normalizedCounts = 100 * counts / sum(counts);
subplot(2,1,2);
bar(binValues, normalizedCounts, 'barwidth', 1,'FaceColor',[.9 .9 .9]);
xlim([-20 20])
xlabel('\textrm{Vy,$\mu$m/min}', 'interpreter', 'latex');
ylabel('\textrm{$\%$}', 'interpreter', 'latex');
ChangeInterpreter(gcf,'Latex');



%% Plot basis functions
%done = plot_field(Theta,Z,knots,grid_limits,basis_type);

% fig5
% Modes and tracks
% figure; set(gcf,'color','w');
% hold on;
% for j = 1:nTracks
%    plot(X_rts{j}(1:end-1,1),X_rts{j}(1:end-1,2),'-b','LineWidth',1); hold on; 
% end
%  for j = 1:nTracks % mark the starting point of each track
%    if Mode{j}(1) == 1
%         plot(X_rts{j}(1:end-1,1),X_rts{j}(1:end-1,2),'-g','LineWidth',1); hold on;
%    elseif Mode{j}(1) == 2
%         plot(X_rts{j}(1:end-1,1),X_rts{j}(1:end-1,2),'-b','LineWidth',1); hold on;
%    else
%         plot(X_rts{j}(1,1),X_rts{j}(1,2),'*r','LineWidth',1); hold on;
%    end
%  end
% hold on;
% text(700, 750,2, '$\star$', 'interpreter', 'latex','Color','b','FontSize',15);
% text(725, 750,2, '- $M^1$', 'interpreter', 'latex','Color','k','FontSize',15);
% text(700, 700,2, '$\star$', 'interpreter', 'latex','Color','g','FontSize',15);
% text(725, 700,2, '- $M^2$', 'interpreter', 'latex','Color','k','FontSize',15);
% hold on;
% xlabel('\textrm{X, px}', 'interpreter', 'latex');
% ylabel('\textrm{Y, px}', 'interpreter', 'latex');
% ChangeInterpreter(gcf,'Latex');
% 
% 
% % fig6
% % Heatmap of chemotactic field
% figure; set(gcf,'color','w');
% done = plot_heatmap(Theta,Z,knots,grid_limits,basis_type);
% hold on;
% xlabel('\textrm{X, px}', 'interpreter', 'latex');
% ylabel('\textrm{Y, px}', 'interpreter', 'latex');
% ChangeInterpreter(gcf,'Latex');

% fig7
%[X_resample,P_resample,nResampled,Grad] = resample_tracks(X_rts,P_rts,Mode,1,ll,nTracks,iter,Z,knots,basis_type);
% Heatmap + tracks
figure; set(gcf,'color','w');
done = plot_heatmap(Theta,Z,knots,grid_limits,basis_type);
hold on;
for j = 1:nTracks
   plot(X_rts{j}(1:end-1,1),X_rts{j}(1:end-1,2),'-k','LineWidth',1); hold on; 
end
% for j = 1:nResampled
%    plot(X_rresample{j}(1:end-1,1),X_resample{j}(1:end-1,2),'-g','LineWidth',1); hold on; 
% end
%  for j = 1:nTracks % mark the starting point of each track
%    if Mode{j}(1) == 1
%         plot(X_rts{j}(1:end-1,1),X_rts{j}(1:end-1,2),'-g','LineWidth',1); hold on;
%    elseif Mode{j}(1) == 2
%         plot(X_rts{j}(1:end-1,1),X_rts{j}(1:end-1,2),'-k','LineWidth',1); hold on;
%    else
%         plot(X_rts{j}(1,1),X_rts{j}(1,2),'*w','LineWidth',1); hold on;
%    end
%  end
% xlabel('\textrm{X, px}', 'interpreter', 'latex');
% ylabel('\textrm{Y, px}', 'interpreter', 'latex');
ChangeInterpreter(gcf,'Latex');

%% Convergence plots

% % fig8 
% % parameter convergence
% figure; set(gcf,'color','w');
% plot(DT_plot,'ko','Linewidth',1);
% ylabel('$\Delta\theta$', 'interpreter', 'latex');
% xlabel('\textrm{Iteration}');
% ChangeInterpreter(gcf,'Latex');

%%
% fig9
% mode switching
iModel = 54
figure; set(gcf,'color','w');
subplot(5,1,1);
Track1 = 8;
stairs(Model_model_all{iModel}{Track1},'r','Linewidth',2); hold on;
% stairs(Mode_imms{Track1},'b','Linewidth',1.5); hold on;
stairs(Mode_all{iModel}{Track1},'k','Linewidth',1);
ylim([0.5;2.5]); xlim([1,inf]); xlabel('time, min'); ylabel('mode');
subplot(5,1,2);
Track2 = 25;
stairs(Model_model_all{iModel}{Track2},'r','Linewidth',2); hold on;
% stairs(Mode_imms{Track2},'b','Linewidth',1.5); hold on;
stairs(Mode_all{iModel}{Track2},'k','Linewidth',1);
ylim([0.5;2.5]); xlim([1,inf]); xlabel('time, min'); ylabel('mode');
subplot(5,1,3);
Track3 = 45;
stairs(Model_model_all{iModel}{Track3},'r','Linewidth',2); hold on;
% stairs(Mode_imms{Track3},'b','Linewidth',1.5); hold on;
stairs(Mode_all{iModel}{Track3},'k','Linewidth',1);
ylim([0.5;2.5]); xlim([1,inf]); xlabel('time, min'); ylabel('mode');
subplot(5,1,4);
Track4 = 71;
stairs(Model_model_all{iModel}{Track4},'r','Linewidth',2); hold on;
% stairs(Mode_imms{Track4},'b','Linewidth',1.5); hold on;
stairs(Mode_all{iModel}{Track4},'k','Linewidth',1);
ylim([0.5;2.5]); xlim([1,inf]); xlabel('time, min'); ylabel('mode');
subplot(5,1,5);
Track5 = 97;
stairs(Model_model_all{iModel}{Track5},'r','Linewidth',2); hold on;
% stairs(Mode_imms{Track5},'b','Linewidth',1.5); hold on;
stairs(Mode_all{iModel}{Track5},'k','Linewidth',1);
ylim([0.5;2.5]); xlim([1,inf]); xlabel('time, sec'); ylabel('mode');
ChangeInterpreter(gcf,'Latex');


%% Monte Carlo results and plots

% Mode transitions;
switches_percent(1:2) = switches(1,:)/switch_from(1);
switches_percent(3:4) = switches(2,:)/switch_from(2);
% cat = categorical({'1 to 1','1 to 2','2 to 1','2 to 2'});
figure; set(gcf,'color','w');
bar(switches_percent,0.5,'FaceColor',[.9 .9 .9]);
set(gca,'XTickLabel',{'1 to 1','1 to 2','2 to 1','2 to 2'})
ChangeInterpreter(gcf,'Latex');

% Field estimation results
% model field
 Theta_model = ones(ll,1);
 Theta_model(1:3,1) = 20;
 Theta_model(4:6,1) = 30;
 Theta_model(7:9,1) = 40;
%  Theta_model(13:16,1) = 50;
 Delta_model = get_deltas(Theta_model,knots);

 % Mean estimate
for i=1:length(Theta)
    Theta_mean(i,1) = mean(Theta_all(i,:));
end

% fig - modeled field
figure; set(gcf,'color','w');
done1 = plot_heatmap(Theta_model,Z,knots,grid_limits,basis_type);
hold on;
side = knots(1,2) - knots(1,1)
for i=1:length(Theta_model)
    a = knots(1,i*2-1);
    b = knots(2,i*2-1);
    p=rectangle('Position',[a,b,side,side],'Curvature',0.2,'EdgeColor','w'); 
    hold on;
end
text(200,270,2, '$\theta_1 = 20$', 'interpreter', 'latex','Color','k','FontSize',14);
text(200,450,2, '$\theta_2 = 20$', 'interpreter', 'latex','Color','k','FontSize',14);
text(200,630,2, '$\theta_3 = 20$', 'interpreter', 'latex','Color','k','FontSize',14);
text(380,270,2, '$\theta_4 = 30$', 'interpreter', 'latex','Color','k','FontSize',14);
text(380,450,2, '$\theta_5 = 30$', 'interpreter', 'latex','Color','k','FontSize',14);
text(380,630,2, '$\theta_6 = 30$', 'interpreter', 'latex','Color','k','FontSize',14);
text(560,270,2, '$\theta_7 = 40$', 'interpreter', 'latex','Color','k','FontSize',14);
text(560,450,2, '$\theta_8 = 40$', 'interpreter', 'latex','Color','k','FontSize',14);
text(560,630,2, '$\theta_9 = 40$', 'interpreter', 'latex','Color','k','FontSize',14);
xlabel('\textrm{X,  $\mu$m}', 'interpreter', 'latex');
ylabel('\textrm{Y,  $\mu$m}', 'interpreter', 'latex');
ChangeInterpreter(gcf,'Latex');

% fig - estimated field (mean)
figure; set(gcf,'color','w');
done2 = plot_heatmap(Theta_mean,Z,knots,grid_limits,basis_type);
hold on;
side = knots(1,2) - knots(1,1);
for i=1:length(Theta_model)
    a = knots(1,i*2-1);
    b = knots(2,i*2-1);
    p=rectangle('Position',[a,b,side,side],'Curvature',0.2,'EdgeColor','w'); 
    hold on;
end
text(200,270,2, ['$\hat{\theta}_1 =$', num2str(Theta_mean(1))], 'interpreter', 'latex','Color','k','FontSize',12);
text(200,450,2, ['$\hat{\theta}_2 =$', num2str(Theta_mean(2))], 'interpreter', 'latex','Color','k','FontSize',12);
text(200,630,2, ['$\hat{\theta}_3 =$', num2str(Theta_mean(3))], 'interpreter', 'latex','Color','k','FontSize',12);
text(380,270,2, ['$\hat{\theta}_4 =$', num2str(Theta_mean(4))], 'interpreter', 'latex','Color','k','FontSize',12);
text(380,450,2, ['$\hat{\theta}_5 =$', num2str(Theta_mean(5))], 'interpreter', 'latex','Color','k','FontSize',12);
text(380,630,2, ['$\hat{\theta}_6 =$', num2str(Theta_mean(6))], 'interpreter', 'latex','Color','k','FontSize',12);
text(560,270,2, ['$\hat{\theta}_7 =$', num2str(Theta_mean(7))], 'interpreter', 'latex','Color','k','FontSize',12);
text(560,450,2, ['$\hat{\theta}_8 =$', num2str(Theta_mean(8))], 'interpreter', 'latex','Color','k','FontSize',12);
text(560,630,2, ['$\hat{\theta}_9 =$', num2str(Theta_mean(9))], 'interpreter', 'latex','Color','k','FontSize',12);
xlabel('\textrm{X,  $\mu$m}', 'interpreter', 'latex');
ylabel('\textrm{Y,  $\mu$m}', 'interpreter', 'latex');
ChangeInterpreter(gcf,'Latex');
% fig - modeled gradient
done3 = plot_gradient(Theta_model,Z,knots,grid_limits,basis_type);
hold on;
xlabel('\textrm{X,  $\mu$m}', 'interpreter', 'latex');
ylabel('\textrm{Y,  $\mu$m}', 'interpreter', 'latex');
ChangeInterpreter(gcf,'Latex');
%fig - estimated gradient
done4 = plot_gradient(Theta_mean,Z,knots,grid_limits,basis_type);
hold on;
xlabel('\textrm{X,  $\mu$m}', 'interpreter', 'latex');
ylabel('\textrm{Y,  $\mu$m}', 'interpreter', 'latex');
ChangeInterpreter(gcf,'Latex');
%% Plot difference between mean estimate and true field
figure; set(gcf,'color','w');
done5 = plot_difference(Theta_mean,Theta_model,Z,knots,grid_limits,basis_type);
hold on;
xlabel('\textrm{X,  $\mu$m}', 'interpreter', 'latex');
ylabel('\textrm{Y,  $\mu$m}', 'interpreter', 'latex');
ChangeInterpreter(gcf,'Latex');

Theta_ones = ones(9,1)
for i=1:length(Theta_all)
    Theta_delta = Theta_all(:,i) - Theta_model;
    scaling_coef(i) = Theta_delta'*Theta_ones*inv(Theta_ones'*Theta_ones);
end
numberOfBins = 20;
[counts, binValues] = hist(scaling_coef, numberOfBins);
normalizedCounts = 100 * counts / sum(counts);
figure; set(gcf,'color','w');
bar(binValues, normalizedCounts, 'barwidth', 1,'FaceColor',[.9 .9 .9]);
xlim([-10 10])
xlabel('c', 'interpreter', 'latex');
ylabel('\textrm{$\%$}', 'interpreter', 'latex');
ChangeInterpreter(gcf,'Latex');
%% Variance ellipsoids (now 3D)

% FIM100 = zeros(length(Theta),length(Theta));
% for iModel=1:100
%     FIM100 = FIM100 + Fisher_all{iModel};
% end

FIM50 = zeros(length(Theta),length(Theta));
for iModel=1:50
    FIM50 = FIM50 + Fisher_all{iModel};
end

FIM10 = zeros(length(Theta),length(Theta));
for iModel=1:10
    FIM10 = FIM10 + Fisher_all{iModel};
end

% for iTheta=1:length(Theta)
%     Theta_mean(iTheta) = mean(Theta_all(:,iTheta));
% end

% Theta_cov = cov(Theta_mean,Theta_mean)


% for iPair=1:2:size(Delta_all,1)-1
%     co{iPair} = cov(Delta_all(iPair,:),Delta_all(iPair+1,:));
%     mea{iPair} = [mean(Delta_all(iPair,:)),mean(Delta_all(iPair+1,:))];
%     ell_points_delta{iPair} = build_ellips(mea{iPair},co{iPair});
% end

% for iPair=1:2:size(Delta_all,1)-1
%     figure;set(gcf,'color','w');
%     plot(Delta_all(iPair,:),Delta_all(iPair+1,:),'*k'); hold on;
%     plot(Delta_model(iPair),Delta_model(iPair+1),'*b','Linewidth',3); hold on;
%     plot(mea{iPair}(1),mea{iPair}(2),'*r','Linewidth',3); hold on;
%     plot(ell_points_delta{iPair}(:,1),ell_points_delta{iPair}(:,2),'r');
%     xlabel(['$\delta_{' num2str(iPair) '}$']);
%     ylabel(['$\delta_{' num2str(iPair+1) '}$']);
%     ChangeInterpreter(gcf,'Latex');
% end

% for iPair=1:6  %2:size(Theta_all,1)-1
%     co_th{iPair} = cov(Theta_all(iPair,:),Theta_all(iPair+3,:));
%     mea_th{iPair} = [Theta_mean(iPair,:),Theta_mean(iPair+3,:)];
%     ell_points_theta{iPair} = build_ellips(mea_th{iPair},co_th{iPair});
% end

    

    
    Theta_hor = [Theta_all(1,:);Theta_all(4,:);Theta_all(7,:)];
    Theta_hor = Theta_hor';
    [X_el,Y_el,Z_el] = build_ellipsoid(Theta_hor);
    
    figure;set(gcf,'color','w');
    plot3(Theta_hor(:,1),Theta_hor(:,2),Theta_hor(:,3),'o','Color',[80/255, 80/255, 80/255]); hold on;
    plot3(Theta_mean(1),Theta_mean(4),Theta_mean(7),'*r','LineWidth',5); hold on;
    plot3(Theta_model(1),Theta_model(4),Theta_model(7),'*k','LineWidth',5); hold on;
    h = surf(X_el,Y_el,Z_el); alpha 0.5
    set(h, 'facecolor',[243/255, 222/255, 187/255]);
    set(h, 'edgecolor',[237/255, 177/255, 32/255])
    grid on;
    xlim([0,40]); ylim([10,50]); zlim([20,60]);
    xlabel(['$\theta_{1}$']);
    ylabel(['$\theta_{4}$']);
    zlabel(['$\theta_{7}$']);
    legend('Estimates','Mean estimate','True value','$95\%$ confidence region')
    legend('Location','northeast')
    ChangeInterpreter(gcf,'Latex');
    


% for iPair=1:6 %2:size(Theta_all,1)-1
%     figure;set(gcf,'color','w');
%     plot(Theta_all(iPair,:),Theta_all(iPair+3,:),'*k','DisplayName','Estimates'); hold on;
%     plot(Theta_model(iPair),Theta_model(iPair+3),'*b','Linewidth',3,'DisplayName','True value'); hold on;
%     plot(mea_th{iPair}(1),mea_th{iPair}(2),'*r','Linewidth',3,'DisplayName','Mean estimte'); hold on;
%     plot(ell_points_theta{iPair}(:,1),ell_points_theta{iPair}(:,2),'r','DisplayName','$95\%$ confidence region');
%     xlabel(['$\theta_{' num2str(iPair) '}$']);
%     ylabel(['$\theta_{' num2str(iPair+3) '}$']);
% %     legend('Estimation results','True value','Mean estimte','$95\%$ confidence region')
%     legend('Location','northwest')
%     ChangeInterpreter(gcf,'Latex');
% end
%% Convergence results
figure;set(gcf,'color','w');
for iModel=1:nModels
    A = dTheta_all{iModel}';
    plot(A); hold on;
end
plot([1,10],[0.001,0.001],'k');
xlabel(['Iteration']);
ylabel(['$\Delta\theta$']);
set(gca, 'YScale', 'log');
ChangeInterpreter(gcf,'Latex');

%% Save results
% 
% filename = ['estimation_reusults_colorednoise' num2str(iModel)];
% save(filename,'dTheta_all','Theta_all','Theta_model','Mode_all','Model_model_all','Q','R','Fisher_all');
% 
%% Making tikz
% cleanfigure;
% matlab2tikz('scaling_coef.tikz', 'showInfo', false, ...
%         'parseStrings',false,'standalone', false, ...
%         'height', '\figureheight', 'width','\figurewidth');
