% ------------ Code Descriptions ------------
% This is the code for running multiple Monte Carlo simulations of ASL process 
% under stationary environments in Figs. 10-12.

% Input:
% -model: parameters for the social learning task
% -delta: step-size
% -T:     terminal time
% -Tm:    number of Monte Carlo runs

% Output:
% -error: instantaneous error probability of each agent for different
%         combination policies, averaged over Tm Monte Carlo runs.

function error=Gaussian_tensor(model,delta,T,Tm)
A=model.A;
N=model.N;
M=model.M;
mu=model.mu;
sigma=model.sigma;
sigma_n=model.sigma_n;
NN=size(A,3);
AA=[];
for i=1:NN
    AA=blkdiag(AA,A(:,:,i));
end
%% learning process: diffusion 
Lambda=zeros(NN*N,Tm,M-1);
error=zeros(NN*N,T);
X=zeros(Tm,N,M-1);
% updating rule
for t=1:T
    Data=mvnrnd(mu(:,1),diag(sigma+sigma_n),Tm); 
    X(:,:,1)=transpose((mu(:,1)-mu(:,2))./sigma).*(Data-transpose((mu(:,2)+mu(:,1)))/2);
    X(:,:,2)=transpose((mu(:,1)-mu(:,3))./sigma).*(Data-transpose((mu(:,3)+mu(:,1)))/2);
    for m=1:M-1
        Lambda(:,:,m)=AA'*((1-delta)*Lambda(:,:,m)+delta*transpose(repmat(X(:,:,m),1,NN)));
    end
    aux=(Lambda(:,:,1)<=zeros(NN*N,Tm))+(Lambda(:,:,2)<=zeros(NN*N,Tm));
    error(:,t)=transpose(sum(transpose(aux>0))/Tm);    
end

