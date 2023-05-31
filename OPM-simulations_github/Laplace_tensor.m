% ------------ Code Descriptions ------------
% This is the code for running multiple Monte Carlo simulations of the ASL 
% process under stationary environments in Figs. 4, 5 and 7.

% This function will be used in Laplace_main.m.

% Input:
% -model: parameters for the social learning task
% -delta: step-size
% -T: terminal time
% -Tm: number of Monte Carlo runs

% Output:
% -error: instantaneous error probability of each agent for different
%         combination policies, averaged over Tm Monte Carlo runs.

function error=Laplace_tensor(model,delta,T,Tm)
A=model.A;
N=model.N;
M=model.M;
mu=model.mu;
sigma=model.sigma;
NN=size(A,3);
AA=[];
for i=1:NN
    AA=blkdiag(AA,A(:,:,i));
end

%% learning process: diffusion 
Lambda=zeros(NN*N,Tm,M-1);
error=zeros(NN*N,T);
X=zeros(N,Tm,M-1);
% updating rule
for t=1:T
    a=rand(N,Tm)-0.5;
    Data=mu(:,1)-sigma(:,1).*sign(a).*log(1-2*abs(a));
    X(:,:,1)=log(sigma(:,2)./sigma(:,1))+abs(Data-mu(:,2))./sigma(:,2)-abs(Data-mu(:,1))./sigma(:,1);
    X(:,:,2)=log(sigma(:,3)./sigma(:,1))+abs(Data-mu(:,3))./sigma(:,3)-abs(Data-mu(:,1))./sigma(:,1);
    for m=1:M-1
        Lambda(:,:,m)=AA'*((1-delta)*Lambda(:,:,m)+delta*repmat(X(:,:,m),NN,1));
    end
    aux=(Lambda(:,:,1)<=zeros(NN*N,Tm))+(Lambda(:,:,2)<=zeros(NN*N,Tm));
    error(:,t)=transpose(sum(transpose(aux>0))/Tm);
end