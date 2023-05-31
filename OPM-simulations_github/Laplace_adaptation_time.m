% ------------ Code Descriptions ------------
% This is the code for Fig. 7: adaptation time 
% we use the data with seed=1015 and 5000000 Monte Carlo runs

load('final_data/Laplace_steady_delta=0.01_seed=1015.mat')

model=load('A_Laplace.mat');
N=model.N;
M=model.M;
A_N=model.A_N;
mu=model.mu;
sigma=model.sigma;
Perron_eigenvector=model.Perron_eigenvector;

%% --------- Tstar and PHI --------
PHI=zeros(A_N,M-1);Tstar=zeros(A_N,M-1);
tt=1/100;
for i=1:A_N
    for m=1:M-1
        for t=-tt:-tt:-200
            LAMBDA1=Laplace_LMGF(N,mu(:,1+m),mu(1,1),t,Perron_eigenvector(:,i)); 
            PHI(i,m)=PHI(i,m)+tt*LAMBDA1/t;
            if LAMBDA1>0
               Tstar(i,m)=t+tt;
               break
            end
        end
    end
end

%% --------- Approximations for Tstar and PHI --------
dl=zeros(N,2);
Var=zeros(N,2);
for i=1:N
    [a1,a2]=Laplace_moments(mu(i,1),mu(i,2),sigma(i,1),sigma(i,2));
    dl(i,1)=a1;
    Var(i,1)=a2;  
    [a1,a2]=Laplace_moments(mu(i,1),mu(i,3),sigma(i,1),sigma(i,3));
    dl(i,2)=a1;
    Var(i,2)=a2;      
end
Mave=Perron_eigenvector'*dl;
Varave=transpose(Perron_eigenvector.^2)*Var;

Tstar_approx = - 2 * Mave ./ Varave;
PHI_approx = - Mave.^2 ./ Varave;

%% ------- adaptation time comparison -------
Index=1;
Epsilon=exp(log(0.01):(log(0.9)-log(0.01))/29:log(0.9));

% -- T_adap defined in our paper
T_adap=zeros(length(Epsilon),1);
for i=1:length(Epsilon)
    T_adap(i)=log(1-sqrt(1-Epsilon(i)))/log(1-delta(Index));
end

% -- T_ASL given by (76) in the ASL paper.
T_ASL=zeros(length(Epsilon),A_N); 
K1=abs(Tstar.*Mave);K=max(transpose(K1));

for i=1:length(Epsilon)
    for j=1:A_N
        T_ASL(i,j)=log(K(j)/(Epsilon(i)*PHI(j,2)))/log(1/(1-delta(Index)));
    end
end

% -- simulated adaptation time
T_sim=zeros(length(Epsilon),A_N);
for i=1:length(Epsilon)
    c=Error_ave(:,:,Index)./(Error_steady_ave(:,Index).^(1-Epsilon(i)));
    for j=1:A_N
        p=0;flag=0;
        while flag==0
            p=p+0.001;
            ii=find(c(j,:)>(1-p) & c(j,:)<=(1+p));
            if ~isempty(ii)
                for iii=1:length(ii)
                    if max(c(j,ii(iii):end))<=(1) 
                        T_sim(i,j)=ii(iii);
                        flag=1;
                        break;
                    end
                end
            end
        end
    end
end

% -- figure
figure;hold on;box on;grid minor
set(gcf, 'DefaultTextInterpreter', 'latex')
set(gca,'XScale','log','Fontname','Times New Roman','Fontsize',20);
xlabel('$\omega$');ylabel('Adaptation time')
grey = [0.7, 0.7, 0.7];
for i=1:4
    plot(Epsilon,T_sim(:,i),'b-o','Linewidth',1,'Markersize',7,'MarkerFaceColor','b');
    plot(Epsilon,T_ASL(:,i),'--','Color',grey,'Linewidth',1,'Markersize',8)
    plot(Epsilon,T_sim(:,i+5),'g-o','Linewidth',1,'Markersize',7,'MarkerFaceColor','g');
    plot(Epsilon,T_ASL(:,i+5),'--','Color',grey,'Linewidth',1,'Markersize',8)
end
h1=plot(Epsilon,T_sim(:,5),'b-o','Linewidth',1,'Markersize',7,'MarkerFaceColor','b');
h2=plot(Epsilon,T_sim(:,10),'g-o','Linewidth',1,'Markersize',7,'MarkerFaceColor','g');
h3=plot(Epsilon,T_adap,'r-.','Linewidth',1);
h4=plot(Epsilon,T_ASL(:,10),'--','Color',grey,'Linewidth',1,'Markersize',8);
legend([h1,h2,h3,h4],'left-stochastic matrix','doubly-stocahstic matrix', ...
    '${\sf T_{adap}}(\omega)$','${\sf T_{ASL}}(\pi,\omega)$','Interpreter','latex','NumColumns',1)
exportgraphics(gcf, 'final_figs/Laplace_adaptation_time.pdf','ContentType','vector')
