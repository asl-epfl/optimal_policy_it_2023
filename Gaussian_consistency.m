% --------------------------- Code Descriptions ---------------------------
% This is the code for Fig. 8: evolution of log-belief ratios.
% -------------------------------------------------------------------------

T=2000;Tm=1;delta=0.002;
model=load('A_noisy_Gaussian.mat');

seed=12; % random seed for one realization

LAMBDA=consistency(seed,model,T,Tm,delta);

m_ave=weightedKLdivergence(model);

plot_consistency(T,m_ave,LAMBDA)


% --- functions ---
function LAMBDA=consistency(seed,model,T,Tm,delta) 
A=model.A;
N=model.N;
M=model.M;
mu=model.mu;
sigma=model.sigma;
sigma_n=model.sigma_n;
AA=A(:,:,1);
NN=1;
LAMBDA=zeros(NN*N,T,M-1);
Lambda=zeros(NN*N,Tm,M-1);
X=zeros(Tm,N,M-1);
rng(seed)

% learning process: diffusion
for t=1:T
    Data=mvnrnd(mu(:,1),diag(sigma+sigma_n),Tm);
    X(:,:,1)=transpose((mu(:,1)-mu(:,2))./sigma).*(Data-transpose((mu(:,2)+mu(:,1)))/2);
    X(:,:,2)=transpose((mu(:,1)-mu(:,3))./sigma).*(Data-transpose((mu(:,3)+mu(:,1)))/2);
    for m=1:M-1
        Lambda(:,:,m)=AA'*((1-delta)*Lambda(:,:,m)+delta*transpose(repmat(X(:,:,m),1,NN)));
    end
    LAMBDA(:,t,:)=sum(Lambda,2)./Tm;  
end
end

function m_ave=weightedKLdivergence(model)
N=model.N;
M=model.M;
mu=model.mu;
sigma=model.sigma;
sigma_n=model.sigma_n;
Perron_eigenvector=model.Perron_eigenvector;
Truth=1;
KL=zeros(N,M);
KL_n=zeros(N,M-1);
n0=5000;
N0=-10:1/n0:10;
for i=1:N
    a=normpdf(N0,mu(i,Truth),sqrt(sigma(i)+sigma_n(i)));
    for m=1:M
        b=normpdf(N0,mu(i,m),sqrt(sigma(i)));
        KL(i,m)=a*transpose(log(a./b))/n0;
    end
    b1=normpdf(N0,mu(i,1),sqrt(sigma(i)));
    for m=1:M-1
        b2=normpdf(N0,mu(i,m+1),sqrt(sigma(i)));
        KL_n(i,m)=a*transpose(log(b1./b2))/n0;
    end
end
m_ave=transpose(Perron_eigenvector(:,1))*KL_n;
end

function plot_consistency(T,m_ave,LAMBDA)
figure;hold on;box on;grid on
set(gcf,'DefaultTextInterpreter','latex');
xlabel('$i$','Interpreter','latex');
ylabel('Log-belief ratio')
set(gca,'Yscale','linear','Fontname','Times New Roman','Fontsize',20, ...
    'defaultTextInterpreter','latex');
for i=1:10
    h1=plot(1:T,LAMBDA(i,:,1),'b-','LineWidth',1.1);
    h2=plot(1:T,LAMBDA(i,:,2),'g-','LineWidth',1.1);
end
h3=plot(1:T,m_ave(1)*ones(T,1),'r--','linewidth',1.2);
plot(1:T,m_ave(2)*ones(T,1),'r--','linewidth',1.2);
legend([h1,h2,h3],'$\boldmath{\lambda}_{k,i}(\theta_2)$','$\boldmath{\lambda}_{k,i}(\theta_3)$', ...
    '$\sf{m_{ave}}(\pi^{\sf{u}},\theta)$','Interpreter','latex','Orientation','horizontal','Location','best')
exportgraphics(gcf,'final_figs/Gaussian_consistency.pdf','ContentType','vector')
end
