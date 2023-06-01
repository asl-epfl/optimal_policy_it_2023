% --------------------------- Code Descriptions ---------------------------
% This is the code for generating the 10 combination matrices used in
% Section V. A.

% The resulting matrices are included in the model data A_Laplace.m.
% -------------------------------------------------------------------------

Is_symmetric=0;
p=0.5;
rng(1);
A0 = rand(N) + eye(N);
A0 = (A0>p);
A0 = A0 + Is_symmetric*A0'; % asymmetric: directed graph
A0 = sign(A0)./sum(sign(A0)); % averaging rule
A_sign=sign(A0);
A_N=10;
A=zeros(N,N,A_N);
A(:,:,1)=A0;kk=1;

%% generate five left-stochastic combination matrices 
for k=1:A_N-6
    AA=rand(N).*A_sign;
    AA=AA./sum(AA);
    A(:,:,k+1)=AA;
    kk=kk+1;
end

%% generate five doubly-stochastic matrices 
for k=1:5
    A(:,:,kk+1)=createDoublyStochasticMatrix(N,A_sign,0);
    kk=kk+1;
end

%% calculate the Perron eigenvectors
Perron_eigenvector=zeros(N,A_N);
for i=1:A_N
    [V,D]=eig(A(:,:,i));
    [B,I]=sort(abs(D*ones(N,1)));
    Perron_eigenvector(:,i)=abs(V(:,I(end)))./abs(sum(V(:,I(end))));
end