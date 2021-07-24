function Y = lle_lite(X, k, d)
%% minifold learning: locally linear embedding
% k : number of neighbors
% X : data as D x N matrix (D = dimensionality, N = #points) 
% d : low dimension;

[D, N] = size(X); 
I = ones(k,1); 
W = zeros(N,N);

if(k>D) % regularlizer in case constrained fits are ill conditioned
  lambda=1e-3; 
else
  lambda=0;
end

%% find k neighbors && reconstrust weight W
for i =1 : N
    X_i = repmat(X(:,i), 1, N);
    diff = X_i - X;
    dist = sum(diff.^2);
    [~, Xsort] = sort(dist);
    neigh_index(:,i) = Xsort(2 : k+1 )'; 
    % weight W
    diff_neigh = repmat(X(:,i),1,k) - X(:,neigh_index(:,i)); %X_i - X_j
    C = diff_neigh' * diff_neigh; %k * k 
    C = C + eye(k,k)* lambda*trace(C);%* trace(C)
    w = C\I;  %(C)^-1 * I, w is molecular
    w2 = I'*w; %w2 = denominator
    W(neigh_index(:,i),i)= w/w2;
end

%% use W reconstrust matrix M=(I-W)'(I-W)

W =sparse(W);
I = eye(N);
M = (I - W) * (I - W)';
[eigenvector, eigenvalue] = eig(M);
eigenvalue = diag(eigenvalue);
[~,pos] = sort(eigenvalue);
M_value_index = pos(1: d+1); %output min value 

%% find eigenvalue = 0,delete eigenvector
tran = eigenvector(:, M_value_index);
thr = 1e-3;
sum_tr = sum(tran);
index = find(sum_tr > thr);
if numel(index) == d+1
   Y = tran(:,1:d)';
else
   tran(:,index)=[];
   Y = tran';
end

% p =sum(tran.*tran);
% j = find(p == min(p));
% tran(:, j) = [];
% Y = tran;
% 
