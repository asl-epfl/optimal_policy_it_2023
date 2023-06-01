% --------------------------- Code Descriptions ---------------------------
% This is the code for Fig. 9: estimation of $t_k^{nc}(\theta)$ under
% the direct and indirect estimation methods.
% -------------------------------------------------------------------------

model=load('A_noisy_Gaussian.mat');

T=10000;
dT=500;
seed=24;

% calculate the estimated values and plot the figure
[T_esm_dr,T_esm_idr,T_true]=MGF_estimation(seed,model,T,dT);

plot_GMF_approx(dT,T,T_esm_dr,T_esm_idr,T_true)


% ---- functions ---

function [T_esm_dr,T_esm_idr,T_true]=MGF_estimation(seed,model,T,dT)
% This funcation calculates the estimated value under two methods;

% Input:
% -seed:  seed for random number generator.
% -model: parameters for the social learning task
% -T:     terminal time
% -dT:    interval of checkpoints

% Output:
% -T_esm_dir: estimated values under the direct estimation method
% -T_esm_idr: estimated values under the indirect estimation method
% -T_true:    theoretical values of $t_k^{nc}(\theta)$

N=model.N;
M=model.M;
mu=model.mu;
sigma=model.sigma;
sigma_n=model.sigma_n;

X=zeros(N,M-1);
tt=1/100; 
TT = -tt : -tt : -2;
MGF = zeros(N, length(TT), 2);
Mean = zeros(N, 1);
Variance = zeros(N, 1);
X_sqr = zeros(N, 1);

T_esm_dr=zeros(N,floor(T/dT),2);
T_esm_idr=zeros(N,floor(T/dT));
rng(seed)

for t=1:T
    Data=mvnrnd(mu(:,1),diag(sigma+sigma_n),1); 
    X(:,1)=transpose((mu(:,1)-mu(:,2))./sigma).*(Data-transpose((mu(:,2)+mu(:,1)))/2);
    X(:,2)=transpose((mu(:,1)-mu(:,3))./sigma).*(Data-transpose((mu(:,3)+mu(:,1)))/2);
    for i=1:N
        for j=1:M-1
            MGF(i,:,j)=exp(X(i,j)*TT)/t+(t-1)*MGF(i,:,j)/t;
            if mod(t,dT)==0
                k=t/dT;
                dis = abs(MGF(i,:,j)-1);
                [~,idx] = min(dis);
                T_esm_dr(i,k,j)=TT(idx);
            end
        end
    end

    X_sqr = X_sqr + transpose(Data.^2);
    Mean = 1/t * transpose(Data) + (t-1)/t * Mean;
    if t>1
        Variance = (X_sqr - t*Mean.^2)/(t-1);
    end

    if mod(t,dT)==0
        T_esm_idr(:,(t/dT)) = - sigma ./ Variance;
    end
end
T_true = - sigma ./ (sigma + sigma_n);
end

function plot_GMF_approx(dT,T,T_esm_dr,T_esm_idr,T_true)
figure('Units','inches','Position',[0,0,8,10]);

set(gcf, 'DefaultTextInterpreter', 'latex')
blue=[0 0.4470 0.7410];
green=[0.4660 0.6740 0.1880];
orange=[0.9290 0.6940 0.1250];

subplot(3,1,1);hold on;box on;grid minor
set(gca,'Fontname','Times New Roman','Fontsize',24);
xlim([dT,T]);ylim([-2,0])
xticks(2000:2000:T)
xlabel('$i$')
title('Agents 1--3','FontWeight','normal')
plot(dT:dT:T,T_esm_dr(1:3,:,1),'o-.','Color',blue,'MarkerSize',10,'MarkerFaceColor','w','LineWidth',1.1);
plot(dT:dT:T,T_esm_idr(1:3,:),'d','Color','m','MarkerSize',10,'MarkerFaceColor','w','LineWidth',1.1);
yline(T_true(1),'r--','LineWidth',1.5);

subplot(3,1,2);hold on;box on;grid minor
set(gca,'Fontname','Times New Roman','Fontsize',24);
xlim([dT,T]);ylim([-2,0])
xlabel('$i$')
xticks(2000:2000:T)
title('Agents 4--7','FontWeight','normal','Interpreter','latex')
plot(dT:dT:T,T_esm_dr(4:7,:,1),'o-.','Color',green,'MarkerSize',10,'MarkerFaceColor','w','LineWidth',1.1);
plot(dT:dT:T,T_esm_idr(4:7,:),'d','Color','m','MarkerSize',10,'MarkerFaceColor','w','LineWidth',1.1);
yline(T_true(4),'r--','LineWidth',1.5);

subplot(3,1,3);hold on;box on;grid minor;
set(gca,'FontName','Times New Roman','Fontsize',24);
xlim([dT,T]);
ylim([-1.5,-0.5]);
xlabel('$i$')
xticks(2000:2000:T)
title('Agents 8--10','FontWeight','normal')
h1=plot(dT,T_esm_dr(10,1,2),'o-.','Color','k','MarkerSize',10,'MarkerFaceColor','w','LineWidth',1.1);
plot(dT:dT:T,T_esm_dr(8:10,:,2),'o-.','Color',orange,'MarkerSize',10,'MarkerFaceColor','w','LineWidth',1.1);
plot(dT:dT:T,T_esm_idr(8:10,:),'d','Color','m','MarkerSize',10,'MarkerFaceColor','w','LineWidth',1.1);
yline(T_true(8),'r--','LineWidth',1.5);
h2=plot(dT:dT:T,T_esm_idr(10,:),'d','Color','m','MarkerSize',10,'MarkerFaceColor','w','LineWidth',1.1);
legend([h1,h2],'direct estimation','indirect estimation','NumColumns',2)
exportgraphics(gcf,'final_figs/Gaussian_MGF_approximation.pdf','ContentType','vector')
end
