my_init
my_map = jet(256);
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
sig2_R = 2;
R = sig2_R*eye(2);
%% Set up Markov chains

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

j=1;
for sig2_Q = [0.5 2 4 8]
    % change Q 
mm = sig2_Q*G_cv*G_cv';
dd = diag(mm);
Q_cv = diag(dd)
Q_cv_inv = (1/sig2_Q)*eye(2);

Q{1} = Q_cv;
Sig_w{1} = G{1}*(inv(G{1}'*G{1}))'*Q_cv_inv*inv(G{1}'*G{1})*G{1}';

Theta = initiate_field(0,60,grid_limits1,knots); %fit splines to a surface
Theta_old = Theta;

models = [1:100];
for iModel = models
clear X Y Mode_model
load(['one_mode_simulated_tracks_' num2str(iModel)]);
Tracks = [1:100]; 
% Tracks = 1; % for now
converged = false;
iter_max  = 5;
iter      = 0;
Theta = zeros(ll,1); 
for k=Tracks
    Z_temp{k}        = zeros(4,length(Y{k}));
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
    x_f{t} = zeros(4,1);
    P_f{t} = eye(4);
end
x_f{1} = X{k}(:,1);
X_out{k}(:,1)   = x_f{1}; 
%% recurtion cycle for KAlman filter
for t=2:T
    % Mode conditioned calculations  
    % Linear
%        clear beta;
%        beta = field_gradient(Z_temp{k}(1:2,t-1),Z,knots,basis_type);
%        u{t-1} = mu_field*beta*Theta;
%        [x_f{t},P_f{t},lik{t}] = kf(y{t},x_f{t-1},u{t-1},P_f{t-1},F{j},B{j},C,Q{j},R);
       % Nonlinear
         [x_f{t},P_f{t},lik{t}] = ekf(y{t},x_f{t-1},P_f{t-1},Q{j},R,F{j},B{j},C,Theta,Z,knots,basis_type);
%         [x_pr,x_f{t},P_pr,P_f{t},lik{t}] = ukf(y{t},x_f{t-1},P_f{t-1},Q{j},R,F{j},B{j},C,Theta,Z,knots,basis_type);            
        X_out{k}(:,t)   = x_f{t};    % state
        P_out{k}(:,:,t) = P_f{t};  % merged covariance
        sigma(:,:) = diag(P_out{k}(:,:,t));
        sig_out{k}(:,t) = sqrt(sigma);
end
%% Recurtion cycle for RTS smoother
% Initialise smoother
x_s{T} = x_f{T};
P_s{T} = P_f{T};
X_sm{k}(:,T) = x_f{T};
P_sm{k}(:,:,T) = P_f{T};
% smoothed variance for plot
sigma(:,:) = diag(P_f{T});
sig_sm{k}(:,T) = sqrt(sigma);
for t=T-1:-1:1 
    % Linear
%        beta = field_gradient(Z_temp{k}(1:2,t+1),Z,knots,basis_type);
%        u{t} = mu_field*beta*Theta;
%        [x_s{t},P_s{t},lik_s{t}] =       rts(x_s{t+1},x_f{t},u{t},P_s{t+1},P_f{t},F{j},B{j},Q{j});
        [x_s{t},P_s{t},lik_s{t}] = erts(x_s{t+1},x_f{t},P_s{t+1},P_f{t},F{j},B{j},Q{j},Theta,Z,knots,basis_type); % ,lik_s{t,j}
%           [x_s{t},P_s{t},lik_s{t}] = urts(x_s{t+1},x_f{t},P_s{t+1},P_f{t},F{j},B{j},G{j},Q{j},Theta,Z,knots,basis_type);       
       % smoothed state for plot
       X_sm{k}(:,t) = x_s{t};
       P_sm{k}(:,:,t) = P_s{t};
       % smoothed variance for plot
       sigma(:,:) = diag(P_sm{k}(:,:,t));
       sig_sm{k}(:,t) = sqrt(sigma);
       % output to be used in the maximisation
      
end % for smoother (t)
Z_temp{k} = X_sm{k};       

end % for track (k)

%% Maximization step [2]
fprintf('M-step... \n')
sum1 = 0;
sum2 = 0;
sum3 = 0;
for k=Tracks
      clear dx x1 x2
      [nn,T] = size(X_sm{k}); 
      x = X_sm{k};
      % CV mode
      x1(:,:) =  X_sm{k};
      dx(:,1:T-1) = x1(:,2:T) - F_cv*x(:,1:T-1);
      dx(:,T) = dx(:,T-1);
     for t=1:T-1
         gg(:,:) = field_gradient(Z_temp{k}(1:2,t),Z,knots,basis_type);
         bb = B{j}*mu_field*gg;
         sum1 = sum1 + bb'*Sig_w{j}*bb;
         sum2 = sum2 + bb'*Sig_w{j}*dx(:,t);
         sum3 = sum3 + bb'*Sig_w{j}*bb;
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

% Save model paremeters
Theta_all(:,iModel) = Theta;
dTheta_all{iModel} = DT_plot;
Fisher_info{iModel} = Expected_info;
Miss_fisher{iModel} = Missing_info;
end % for model (iModel)

 %% Mean estimate
for i=1:length(Theta)
    Theta_mean(i,1) = mean(Theta_all(i,:));
    Std_dev(i,1) = std(Theta_all(i,:));
end
Theta_bias = 100*(Theta_model - Theta_mean)./Theta_model;
Theta_cv   = 100*Std_dev./Theta_mean
Theta_diff = Theta_model - Theta_mean;

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

 %% Plot ellipsoids
% i1 = 1;
% i2 = 2;
% i3 = 3;
% 
%     Theta_hor = [Theta_all(i1,:);Theta_all(i2,:);Theta_all(i3,:)];
%     Theta_hor = Theta_hor';
%     covar(:,1) = COV100([i1,i2,i3],i1);
%     covar(:,2) = COV100([i1,i2,i3],i2);
%     covar(:,3) = COV100([i1,i2,i3],i3);
%     [X_el,Y_el,Z_el] = build_ellipsoid(Theta_hor,covar);
%     [X_el1,Y_el1,Z_el1] = build_ellipsoid_from_data(Theta_hor);
%     figure;
%     plot3(Theta_hor(:,1),Theta_hor(:,2),Theta_hor(:,3),'o','Color',[80/255, 80/255, 80/255]); hold on;
%     plot3(Theta_mean(i1),Theta_mean(i2),Theta_mean(i3),'*r','LineWidth',5); hold on;
%     plot3(Theta_model(i1),Theta_model(i2),Theta_model(i3),'*k','LineWidth',5); hold on;
%     h = surf(X_el,Y_el,Z_el); alpha 0.8   
%     set(h, 'facecolor',[243/255, 222/255, 187/255]);
%     set(h, 'edgecolor',[237/255, 177/255, 32/255]);
%     hold on;
%     h1 = surf(X_el1,Y_el1,Z_el1); alpha 0.5    
%     set(h1, 'facecolor',[187/255, 222/255, 243/255]);
%     set(h1, 'edgecolor',[32/255, 177/255, 237/255]);
%     grid on;
%     view(45, 25);
% %     xlim([-100,150]); ylim([-100,150]); zlim([-100,150]);
%     xlabel(['$\theta_{' num2str(i1) '}$']);
%     ylabel(['$\theta_{' num2str(i2) '}$']);
%     zlabel(['$\theta_{' num2str(i3) '}$']);
%     legend('Estimates','Mean estimate','True value','$95\%$ confidence region','$95\%$ covariance ellipsoid')
%     legend('Location','northeast')
% 	cleanfigure;
% % matlab2tikz('ellipses_100_the123.tikz', 'showInfo', false,'parseStrings',false, ...
% %          'standalone', false,'height', '3cm', 'width','4cm');    
% %%    
% Theta_hor = [Theta_all(i1,1:50);Theta_all(i2,1:50);Theta_all(i3,1:50)];
%     Theta_hor = Theta_hor';
%     covar(:,1) = COV50([i1,i2,i3],i1);
%     covar(:,2) = COV50([i1,i2,i3],i2);
%     covar(:,3) = COV50([i1,i2,i3],i3);
%     [X_el,Y_el,Z_el] = build_ellipsoid(Theta_hor,covar);
%     [X_el1,Y_el1,Z_el1] = build_ellipsoid_from_data(Theta_hor);
%     figure;
%     plot3(Theta_hor(:,1),Theta_hor(:,2),Theta_hor(:,3),'o','Color',[80/255, 80/255, 80/255]); hold on;
%     plot3(Theta_mean_50(i1),Theta_mean_50(i2),Theta_mean_50(i3),'*r','LineWidth',5); hold on;
%     plot3(Theta_model(i1),Theta_model(i2),Theta_model(i3),'*k','LineWidth',5); hold on;
%     h = surf(X_el,Y_el,Z_el); alpha 0.6    
%     set(h, 'facecolor',[243/255, 222/255, 187/255]);
%     set(h, 'edgecolor',[237/255, 177/255, 32/255]);
%     hold on;
%     h1 = surf(X_el1,Y_el1,Z_el1); alpha 0.3    
%     set(h1, 'facecolor',[187/255, 222/255, 243/255]);
%     set(h1, 'edgecolor',[32/255, 177/255, 237/255]);
%     grid on;
%     view(45, 25);
% %     xlim([-100,150]); ylim([-100,150]); zlim([-100,150]);
%     xlabel(['$\theta_{' num2str(i1) '}$']);
%     ylabel(['$\theta_{' num2str(i2) '}$']);
%     zlabel(['$\theta_{' num2str(i3) '}$']);
%     legend('Estimates','Mean estimate','True value','$95\%$ confidence region','$95\%$ covariance ellipsoid')
%     legend('Location','northeast')
% cleanfigure;
% % matlab2tikz('ellipse_50_the123.tikz', 'showInfo', false,'parseStrings',false, ...
% %          'standalone', false,'height', '3cm', 'width','4cm');
% % 
% %%    
% Theta_hor = [Theta_all(i1,1:25);Theta_all(i2,1:25);Theta_all(i3,1:25)];
%     Theta_hor = Theta_hor';
%     covar(:,1) = COV25([i1,i2,i3],i1);
%     covar(:,2) = COV25([i1,i2,i3],i2);
%     covar(:,3) = COV25([i1,i2,i3],i3);
%     [X_el,Y_el,Z_el] = build_ellipsoid(Theta_hor,covar);
%     [X_el1,Y_el1,Z_el1] = build_ellipsoid_from_data(Theta_hor);
%     figure;
%     plot3(Theta_hor(:,1),Theta_hor(:,2),Theta_hor(:,3),'o','Color',[80/255, 80/255, 80/255]); hold on;
%     plot3(Theta_mean_25(i1),Theta_mean_25(i2),Theta_mean_25(i3),'*r','LineWidth',5); hold on;
%     plot3(Theta_model(i1),Theta_model(i2),Theta_model(i3),'*k','LineWidth',5); hold on;
%     h = surf(X_el,Y_el,Z_el); alpha 0.8    
%     set(h, 'facecolor',[243/255, 222/255, 187/255]);
%     set(h, 'edgecolor',[237/255, 177/255, 32/255]);
%     hold on;
%     h1 = surf(X_el1,Y_el1,Z_el1); alpha 0.5    
%     set(h1, 'facecolor',[187/255, 222/255, 243/255]);
%     set(h1, 'edgecolor',[32/255, 177/255, 237/255]);
%     grid on;
%     view(45, 25);
% %     xlim([-100,150]); ylim([-100,150]); zlim([-100,150]);
%     xlabel(['$\theta_{' num2str(i1) '}$']);
%     ylabel(['$\theta_{' num2str(i2) '}$']);
%     zlabel(['$\theta_{' num2str(i3) '}$']);
%     legend('Estimates','Mean estimate','True value','$95\%$ confidence region','$95\%$ covariance ellipsoid')
%     legend('Location','northeast')
% cleanfigure;
% % matlab2tikz('ellipses_25_the123.tikz', 'showInfo', false,'parseStrings',false, ...
% %          'standalone', false,'height', '3cm', 'width','4cm');
% %%    
% Theta_hor = [Theta_all(i1,1:10);Theta_all(i2,1:10);Theta_all(i3,1:10)];
%     Theta_hor = Theta_hor';
%     covar(:,1) = COV10([i1,i2,i3],i1);
%     covar(:,2) = COV10([i1,i2,i3],i2);
%     covar(:,3) = COV10([i1,i2,i3],i3);
%     [X_el,Y_el,Z_el] = build_ellipsoid(Theta_hor,covar);
%     [X_el1,Y_el1,Z_el1] = build_ellipsoid_from_data(Theta_hor);
%     figure;
%     plot3(Theta_hor(:,1),Theta_hor(:,2),Theta_hor(:,3),'o','Color',[80/255, 80/255, 80/255]); hold on;
%     plot3(Theta_mean_10(i1),Theta_mean_10(i2),Theta_mean_10(i3),'*r','LineWidth',5); hold on;
%     plot3(Theta_model(i1),Theta_model(i2),Theta_model(i3),'*k','LineWidth',5); hold on;
%     h = surf(X_el,Y_el,Z_el); alpha 0.8    
%     set(h, 'facecolor',[243/255, 222/255, 187/255]);
%     set(h, 'edgecolor',[237/255, 177/255, 32/255]);
%     hold on;
%     h1 = surf(X_el1,Y_el1,Z_el1); alpha 0.5    
%     set(h1, 'facecolor',[187/255, 222/255, 243/255]);
%     set(h1, 'edgecolor',[32/255, 177/255, 237/255]);
%     grid on;
%     view(45, 25);
% %     xlim([-100,150]); ylim([-100,150]); zlim([-100,150]);
%     xlabel(['$\theta_{' num2str(i1) '}$']);
%     ylabel(['$\theta_{' num2str(i2) '}$']);
%     zlabel(['$\theta_{' num2str(i3) '}$']);
%     legend('Estimates','Mean estimate','True value','$95\%$ confidence region','$95\%$ covariance ellipsoid')
%     legend('Location','northeast')
%     cleanfigure;
% matlab2tikz('ellipses_10_the123.tikz', 'showInfo', false,'parseStrings',false, ...
%          'standalone', false,'height', '3cm', 'width','4cm');
%% Compare mode conditioned estimates
% k = 2;
% figure('Name','SRKF + SRRTS','NumberTitle','off'); 
% set(gcf,'color','w');
% suptitle('Estimation from the measurements on all scales')
% yylabels = {'$\hat{s}_{\mathrm{x}}$','$\hat{s}_{\mathrm{y}}$','$\hat{v}_{\mathrm{x}}$','$\hat{v}_{\mathrm{y}}$'}
% T = length(X{k});
% time = [1:T]';
% xxx = [time' fliplr(time')];
% for j = 1:4
% subplot(2,2,j)
% plot(time,X{k}(j,:),'k','LineWidth',2); hold on;
% plot(time,X_out{k}(j,:),':r','LineWidth',3); hold on;
% plot(time,X_sm{k}(j,:),':b','LineWidth',3); hold on;
% plot(time,X_out{k}(j,:)+sig_out{k}(j,:),'--r','LineWidth',1); hold on;
% plot(time,X_out{k}(j,:)-sig_out{k}(j,:),'--r','LineWidth',1); hold on;
% plot(time,X_sm{k}(j,:)+sig_sm{k}(j,:),'--b','LineWidth',1); hold on;
% plot(time,X_sm{k}(j,:)-sig_sm{k}(j,:),'--b','LineWidth',1); hold on;
% yy = [X_out{k}(j,:)+sig_out{k}(j,:) fliplr(X_out{k}(j,:)-sig_out{k}(j,:))];
% yy_sm = [X_sm{k}(j,:)+sig_sm{k}(j,:) fliplr(X_sm{k}(j,:)-sig_sm{k}(j,:))];
% patch(xxx,yy,'r','FaceAlpha',0.2);
% patch(xxx,yy_sm,'b','FaceAlpha',0.1);
% clear yy
% xlim([time(1), time(end)]);
% xlabel('time, s'); ylabel(yylabels(j));
% end
% h1 =  legend('true','filtered','smoothed');
% 
% %% fig - difference
% figure; 
% colormap(my_map);
% [done1,ZZ] = plot_heatmap(Theta_diff,Z,knots,grid_limits,basis_type);
% hold on;
% side = knots(1,2) - knots(1,1);
% for i=1:length(Theta_mean)
%     a = knots(1,i*2-1);
%     b = knots(2,i*2-1);
%     p=rectangle('Position',[a,b,side,side],'Curvature',0.1,'EdgeColor','w'); 
%     hold on;
% end
% xlabel('\textrm{X, a.u.}', 'interpreter', 'latex');
% ylabel('\textrm{Y,  a.u.}', 'interpreter', 'latex');
% set(findall(gcf,'-property','FontSize'),'FontSize',18);
% ChangeInterpreter(gcf,'Latex');
% %% fig - estimated field (modelled)
% figure; 
% colormap(my_map);
% [done2,ZZ] = plot_heatmap(Theta_model,Z,knots,grid_limits,basis_type);
% hold on;
% side = knots(1,2) - knots(1,1);
% for i=1:length(Theta)
%     a = knots(1,i*2-1);
%     b = knots(2,i*2-1);
%     p=rectangle('Position',[a,b,side,side],'Curvature',0.1,'EdgeColor','w'); 
%     hold on;
% end
% xlabel('\textrm{X, a.u.}', 'interpreter', 'latex');
% ylabel('\textrm{Y,  a.u.}', 'interpreter', 'latex');
% set(findall(gcf,'-property','FontSize'),'FontSize',18);
% ChangeInterpreter(gcf,'Latex');
% 
% %% fig - estimated field (mean)
% figure; 
% colormap(my_map);
% [done4,ZZ] = plot_gradient(Theta_mean,Z,knots,grid_limits,basis_type);
% hold on;
% colormap jet
% colorbar
% xlim([0 1000]);
% ylim([0 1000]);
% %  caxis([0 500]);
% xlabel('\textrm{X, a.u.}', 'interpreter', 'latex');
% ylabel('\textrm{Y,  a.u.}', 'interpreter', 'latex');

%% save workspace to file

Filename = ['results_one_mode_ekf_Q',num2str(sig2_Q),'_R',num2str(sig2_R)]

save(Filename)

end