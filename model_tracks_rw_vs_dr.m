clc; clear;
close all;


%% Test of RTS smoother

% Implemented as a one iterarion of EM algorithm
% Expectation stage is implemented as Kalman smoother
% Log Likelihood is computed for Gassian distributions

%References: (put nice citations in)

% [1] G.R. Holmes et al. "The Neutrophils Eye-View: Inference and Visualisation of the Chemoattractant field Driving Cell Chemotaxis", 2012. 
% [2] V. Kadirkamanathan, S. Anderson "Maximum-Likelihood Estimation of
% Delta-Domain Model Parameters from Noisy Output Signals", 2008.
% [3] S. Fioretti,L. Jetto "A new algorithm for the sequentioal estimation
% of the regularization parameterin the spline smoothing problem", 1992.
% [4] RTS smoother from Sensors 2011, 11 (pp 3747-3749).

% Anastasia Kadochnikova, PhD student at ACSE, the University of Sheffield
% 12/11/2015

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
 Theta(1:4,1) = 100;
 Theta(5:8,1) = 200;
 Theta(9:12,1) = 300;
 Theta(13:16,1) = 400;


% Theta(1:3,1) = 20;
% Theta(4:6,1) = 30;
% Theta(7:9,1) = 40;

 
done = plot_field(Theta,Z,knots,grid_limits,'bspline');
 mu = 10;


 %% State-space model parameters

fprintf(1,'Setting model parameters... \n')
T = 1; % sampling time
N  = 2;  % length of state vector

% Transition matrix
I =   eye(2,2);
O = zeros(2,2);

% For RW (with I for second order, with O for first order)
F_rw = [I - thta*T*I]; 
% For CV
F_cv = [I];  
 
% Measurement matrix
H = [I];

% Contrl matrix
B_cv = [T*I];
%  B_cv = [O; I];

B_rw = [O];

% Disturbance matrix matrix
G_cv = [0.5*T^2*I];
% G_rw = [(T*T/2)*I; T*I];
G_rw = [I];

% Disturbance matrix from from [3]
sig1_Q = 3; % RW  disturbance - cell speed
sig2_Q = 1; % CV disturbance - random component of cell acceleration

% For DIFF
Q_rw = sig1_Q*eye(2);  
% For DRIFT
Q_cv = sig2_Q*eye(2);

P = eye(2); 
%P = F_cv'*P*F_cv + Q_cv; 
R = 1*eye(2);

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

%% Modelling cell tracks
for iModel = 1:1
    iModel
nTracks = 20;
Track_length = 100*ones(nTracks,1); % + floor(100*rand(nTracks,1));
sx_init = 600*rand(nTracks,1) + 200;
sy_init = 600*rand(nTracks,1) + 200;
% sx_init = 300*ones(nTracks,1);
% sy_init = 150*ones(nTracks,1);

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
   % initialise track    
   X{j} = zeros(N,Track_length(j));
   X{j}(1,1) = sx_init(j,1);
   X{j}(2,1) = sy_init(j,1);
   Mode{j}(1) = 1; 
   m =  Mode{j}(1);
   % gradient vector
   noise(:,1) = diag(normrnd(0,R));
   grad_model{j} = zeros(Track_length(j),2,ll);
   for i=2:Track_length(j)
       % Simulate state
       clear beta;
       grad_model{j}(i-1,:,:) = field_gradient(X{j}(1:2,i-1),Z,knots,basis_type);
       aa = F{m}*X{j}(:,i-1);
       beta(:,:) = grad_model{j}(i-1,:,:);
       bb{j}(:,i) = beta*Theta;
       noise_x = diag(normrnd(0,Q{m}));
       nois_x{j}(:,i) = noise_x;
       X{j}(:,i) = F{m}*X{j}(:,i-1) + B{m}*mu*bb{j}(:,i) + G{m}*noise_x;
       noise(:,i) = diag(normrnd(0,R));
       if X{j}(1,i) > 900 || X{j}(1,i) < 0 || X{j}(2,i) > 900 || X{j}(2,i) < 0
           break;
       end
       % Simulate mode
       probs = P_tr(m,:);
       r = rand;
       temp = 0;
       for k=1:length(probs)
           if (r <= probs(k) + temp)
               m = k;
               break;
           else
               temp = temp +  probs(k);
           end
       end
      Mode{j}(i) = m; 
   end
   Xx{j}(:,:) = X{j}(:,1:i);
   clear X{j}
   X{j} = Xx{j};
   Y{j} = H*X{j} + noise(:,:);
   clear noise Xx 
end
% 
% j = 7;
% figure;
% plot(X{j}(1,:),X{j}(2,:),'k','LineWidth',2); hold on; 
% for i=1:Track_length(j)-1
%     quiver(X{j}(1,i),X{j}(2,i),100*bb{j}(1,i),100*bb{j}(2,i),'k'); hold on;
% end
% hold on;
% for i=1:Track_length(j)-1
%     quiver(X{j}(1,i),X{j}(2,i),10*X{j}(3,i),10*X{j}(4,i),'r'); hold on;
% end
% hold on;
% for i=1:Track_length(j)-1
%     quiver(X{j}(1,i),X{j}(2,i),10*nois_x{j}(1,i),10*nois_x{j}(2,i),'b'); hold on;
% end
% 
figure;
cols = colormap('jet');
 plotter = plot_heatmap(Theta,Z,knots,grid_limits,basis_type);
 hold on;
for j = 1:nTracks
   txt = num2str(j);
   text(X{j}(1,1)-2,X{j}(2,1), txt,'Color','r','FontSize',15)
   plot(X{j}(1,:),X{j}(2,:),'k','LineWidth',2); hold on; 
   plot(X{j}(1,1),X{j}(2,1),'r','LineWidth',2); hold on;   
   plot(X{j}(1,1),X{j}(2,1),'*r','LineWidth',2); hold on;    
end
grid on; hold off;
xlabel('X'); ylabel('Y');


X_model = X;
Y_model = Y;
clear X Y
for j=1:nTracks
    X{j} = X_model{j};
    Y{j} = Y_model{j}; 
    Mode_model{j} = Mode{j};
end

filename = ['simulated_tracks_2d_state' num2str(iModel)];
save(filename,'X','Y','Mode_model','nTracks');

clear X Y Mode_model nTracks 
end
