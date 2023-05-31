% ------------ Code Descriptions ------------
% This is the code for running multiple Monte Carlo simulations of ASL process 
% under non-stationary environments in Fig. 14.

% Input:
% -model:  parameters for the social learning task
% -delta:  step-size
% -T:      terminal time
% -Tm:     number of Monte Carlo runs
% -T_tran: time instants of the variation of the true state

% Output:
% -error: instantaneous error probability of each agent for different
%         combination policies, averaged over Tm Monte Carlo runs.

function error=Gaussian_tensor_transient_changing(model,delta,T,Tm,T_tran)
A=model.A;
N=model.N;
M=model.M;
mu=model.mu;
sigma=model.sigma;
if isfield(model,'sigma_n1')
    sigma_n1=model.sigma_n1;
    sigma_n2=model.sigma_n2;
    sigma_n3=model.sigma_n3;
else
    sigma_n1=model.sigma_n;
    sigma_n2=model.sigma_n;
    sigma_n3=model.sigma_n;
end
NN=size(A,3);
AA=[];
for i=1:NN
    AA=blkdiag(AA,A(:,:,i));
end
Lambda=zeros(NN*N,Tm,M-1);
error=zeros(NN*N,T);
X=zeros(Tm,N,M-1);
for t=1:T
    if t<T_tran(1)+1
        Data=mvnrnd(mu(:,1),diag(sigma+sigma_n1),Tm);
        X(:,:,1)=transpose((mu(:,1)-mu(:,2))./sigma).*(Data-transpose((mu(:,2)+mu(:,1)))/2);
        X(:,:,2)=transpose((mu(:,1)-mu(:,3))./sigma).*(Data-transpose((mu(:,3)+mu(:,1)))/2);
        for m=1:M-1
            Lambda(:,:,m)=AA'*((1-delta)*Lambda(:,:,m)+delta*transpose(repmat(X(:,:,m),1,NN)));
        end
        aux=(Lambda(:,:,1)<=zeros(NN*N,Tm))+(Lambda(:,:,2)<=zeros(NN*N,Tm));
        error(:,t)=transpose(sum(transpose(aux>0))/Tm);  
    elseif t>T_tran(1)&& t<T_tran(2)+1
        Data=mvnrnd(mu(:,2),diag(sigma+sigma_n2),Tm);
        X(:,:,1)=transpose((mu(:,1)-mu(:,2))./sigma).*(Data-transpose((mu(:,2)+mu(:,1)))/2);
        X(:,:,2)=transpose((mu(:,1)-mu(:,3))./sigma).*(Data-transpose((mu(:,3)+mu(:,1)))/2);
        for m=1:M-1
            Lambda(:,:,m)=AA'*((1-delta)*Lambda(:,:,m)+delta*transpose(repmat(X(:,:,m),1,NN)));
        end
        aux=(Lambda(:,:,1)>=zeros(NN*N,Tm))+((Lambda(:,:,2)-Lambda(:,:,1))<=zeros(NN*N,Tm));
        error(:,t)=transpose(sum(transpose(aux>0))/Tm);    
    else
        Data=mvnrnd(mu(:,3),diag(sigma+sigma_n3),Tm);
        X(:,:,1)=transpose((mu(:,1)-mu(:,2))./sigma).*(Data-transpose((mu(:,2)+mu(:,1)))/2);
        X(:,:,2)=transpose((mu(:,1)-mu(:,3))./sigma).*(Data-transpose((mu(:,3)+mu(:,1)))/2);
        for m=1:M-1
            Lambda(:,:,m)=AA'*((1-delta)*Lambda(:,:,m)+delta*transpose(repmat(X(:,:,m),1,NN)));
        end
        aux=((Lambda(:,:,1)-Lambda(:,:,2))<=zeros(NN*N,Tm))+(Lambda(:,:,2)>=zeros(NN*N,Tm));
        error(:,t)=transpose(sum(transpose(aux>0))/Tm);  
    end
end