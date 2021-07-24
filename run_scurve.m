clc;
close all;
clear;

% S-CURVE DATASET

N=2000;
K=12;
d=2;

% PLOT TRUE MANIFOLD
tt = [-1:0.1:0.5]*pi;
uu = fliplr(tt);
hh = [0:0.1:1]*5;
xx = [cos(tt) -cos(uu)]'*ones(size(hh));
yy = ones(size([tt uu]))'*hh;
zz = [sin(tt) 2-sin(uu)]'*ones(size(hh));
cc = [tt uu]' * ones(size(hh));

subplot(1,3,1);
surf(xx,yy,zz,cc);

% GENERATE SAMPLED DATA

angle = pi*(1.5*rand(1,N/2)-1);
angle_lr = fliplr(angle);
height = 5*rand(1,N);
X = [[cos(angle), -cos(angle_lr)]; height; [ sin(angle), 2-sin(angle_lr)]];

% SCATTERPLOT OF SAMPLED DATA
subplot(1,3,2);
scatter3(X(1,:),X(2,:),X(3,:),12,[angle angle_lr],'+');

%% RUN LLE ALGORITHM
%Y=lle(X,K,d);
Y=lle_lite(X,K,d);
% SCATTERPLOT OF EMBEDDING
figure;
subplot(1,2,1);cla;
scatter(Y(1,:),Y(2,:),12,[angle angle_lr],'+');title('LLE')

%% PCA dimensionality reduction
C = double(X * X');
[V, D] = eig(C);
D = diag(D); % perform PCA on features matrix 
conf.V_pca = V(:, 1:d);
Y = conf.V_pca' * X;
subplot(1,2,2);
scatter(Y(1,:),Y(2,:),12,[angle angle_lr],'+');title('PCA')

