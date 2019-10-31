my_init
%% Define field model parameters

% define each BF widths in spacial direction - module of sigma is connected
% with size of theta
sigma1 = 50;
sigma2 = 50;
% equal sigmas for isotopic basis functions
Z = [sigma1^2 0; 0 sigma2^2];


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
 Theta(5:8,1) = 100;
 Theta(9:12,1) = 190;
 Theta(13:16,1) = 290;
 Theta_model = Theta;


% Theta(1:3,1) = 20;
% Theta(4:6,1) = 30;
% Theta(7:9,1) = 40;

figure; 
colormap(my_map);
done = plot_field(Theta,Z,knots,grid_limits,'bspline');
mu = 1;


 %% State-space model parameters

fprintf(1,'Setting model parameters... \n')
T = 1; % sampling time
N  = 2;  % order of derivatives
thta = 0.1; % reversion to mean in O-U process 
thta1 = 0.7; % reversion to mean in O-U process 

mean_vel = 0; % mu of O-U process

% Transition matrix
I =   eye(2,2);
O = zeros(2,2);

% For RW (with I for second order, with O for first order)
F_rw = [I   T*I;...
        O   I - thta1*T*I]; 
% For CV
F_cv = [I   T*I;...
        O   I- thta*T*I];  %  
 
% Measurement matrix
H = [I O];

% Contrl matrix
B_cv = [T^2/2*I; T*I];
%  B_cv = [O; I];

B_rw = [O; O];

% Disturbance matrix matrix
G_cv = [T^2/2*I; T*I];
% G_rw = [(T*T/2)*I; T*I];
G_rw = [T^2/2*I; T*I];

W = [I; O];

sig1_Q = 1; % RW  disturbance - cell speed
sig2_Q = 2; % CV disturbance - random component of cell acceleration

% For DIFF
Q_rw = sig1_Q*eye(2);  
% For DRIFT
Q_cv = sig2_Q*eye(2);

P = eye(4); 
%P = F_cv'*P*F_cv + Q_cv; 
R = 2*eye(2);

%% Set up Markov chains

n_models = 2;
P_tr = [0.9 0.1;...
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

%%
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

%% Modelling cell tracks
for iModel = 1:100
    iModel
nTracks = 100;
Track_length = 100*ones(nTracks,1); % + floor(100*rand(nTracks,1));
 sx_init = 500*rand(nTracks,1) + 300;
 sy_init = 500*rand(nTracks,1) + 250;
%  sx_init = 450*ones(nTracks,1);
%  sy_init = 450*ones(nTracks,1);
% 
for j=1:nTracks
   % initialise mode
   Mode{j} = zeros(1,Track_length(j));
   r = rand;
   if (r <= .5)
        m = 1;
   else
        m = 2;
   end
   Mode{j}(1) = m;
   chain(1) = m;
   for i=2:Track_length(j)
        this_step_distribution = P_tr(:,chain(i-1));
        cumulative_distribution = cumsum(this_step_distribution);
        r = rand;
        chain(i) = find(cumulative_distribution>r,1); 
        Mode{j}(i) = chain(i);%
   end
   % initialise track    
   X{j} = zeros(4,Track_length(j));
   X{j}(1,1) = sx_init(j,1);
   X{j}(2,1) = sy_init(j,1);
   X{j}(3,1) = 0;
   X{j}(4,1) = 0; 
   % gradient vector
   noise(:,1) = diag(normrnd(0,sqrt(R)));
   grad_model{j} = zeros(Track_length(j),2,ll);
   for i=2:Track_length(j)
       m = Mode{j}(i);
       % Simulate state
       clear beta;
       grad_model{j}(i-1,:,:) = field_gradient(X{j}(1:2,i-1),Z,knots,basis_type);
       aa = F{m}*X{j}(:,i-1);
       beta(:,:) = grad_model{j}(i-1,:,:);
       bb{j}(:,i) = beta*Theta;
       noise_x = mvnrnd([0,0],sqrt(Q{m}))';
       nois_x{j}(:,i) = noise_x;
       X{j}(:,i) = F{m}*X{j}(:,i-1) + B{m}*mu*bb{j}(:,i) + G{m}*noise_x;
       noise(:,i) = mvnrnd([0,0],sqrt(R))';
       if X{j}(1,i) > 900 || X{j}(1,i) < 0 || X{j}(2,i) > 900 || X{j}(2,i) < 0
           break;
       end
       % Simulate mode
%        probs = P_tr(:,m);
%        r = rand;
%        temp = 0;       
%        for k=1:length(probs)
%            if (r <= probs(k) + temp)
%                m = k;
%                break;
%            else
%                temp = temp +  probs(k);
%            end
%        end
%       Mode{j}(i) = chain(i); 
%        m          = chain(i);
   end
   Xx{j}(:,:) = X{j}(:,1:i);
   clear X{j}
   X{j} = Xx{j};
   Y{j} = H*X{j} + noise(:,:);
   clear noise Xx 
end
% 
% %%
% j = 50;
% kk = 6;
%% figure; 
% colormap(my_map);
% plot(X{j}(1,1:kk),X{j}(2,1:kk),'ok','LineWidth',2); hold on; 
% for i=2:kk
%     quiver(X{j}(1,i),X{j}(2,i),1*bb{j}(1,i),1*bb{j}(2,i),'k'); hold on;
% end
% hold on;
% for i=2:kk
%     quiver(X{j}(1,i),X{j}(2,i),1*nois_x{j}(1,i),1*nois_x{j}(2,i),'b'); hold on;
% end
% hold on;
% for i=2:kk
%     quiver(X{j}(1,i),X{j}(2,i),1*X{j}(3,i),1*X{j}(4,i),'r'); hold on;
% end
% xlabel('X, $\mu$m'); ylabel('Y, $\mu$m');
% set(findall(gcf,'-property','FontSize'),'FontSize',18);
% ChangeInterpreter(gcf,'Latex');
% 
% %%
% figure; 
% colormap(my_map);
%  plotter = plot_heatmap(Theta,Z,knots,grid_limits,basis_type);
%  hold on;
% for j = 1:nTracks
%    txt = num2str(j);
%    text(X{j}(1,1)-2,X{j}(2,1), txt,'Color','r','FontSize',15)
%    plot(X{j}(1,:),X{j}(2,:),'k','LineWidth',2); hold on; 
%    plot(X{j}(1,1),X{j}(2,1),'*r','LineWidth',2); hold on;    
% end
% grid on; hold off;
% xlabel('X, $\mu$m'); ylabel('Y, $\mu$m');
% set(findall(gcf,'-property','FontSize'),'FontSize',18);
% ChangeInterpreter(gcf,'Latex');
% 
%%
% figure; 
% colormap(my_map);
% plotter = plot_heatmap(Theta,Z,knots,grid_limits,basis_type);
% hold on;
% for j = 1:nTracks
%    plot(Y{j}(1,:),Y{j}(2,:),'k','LineWidth',2); hold on; 
%    plot(X{j}(1,1),X{j}(2,1),'*r','LineWidth',2); hold on;    
% end
% grid on; hold off;
% xlabel('X'); ylabel('Y');
% tightfig;
% 
X_model = X;
Y_model = Y;
clear X Y
for j=1:nTracks
    X{j} = X_model{j};
    Y{j} = Y_model{j}; 
    Mode_model{j} = Mode{j};
end

filename = ['simulated_tracks_' num2str(iModel)];
save(filename,'X','Y','Mode_model','nTracks','Theta_model');

clear X Y Mode_model nTracks 
end
