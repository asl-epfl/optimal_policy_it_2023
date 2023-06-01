% --------------------------- Code Descriptions ---------------------------
% This is the main function for generating figures in Section V.A.
% -------------------------------------------------------------------------

% -- create folders for saving data and figs

data_folder_name = 'final_data';
if not(isfolder(data_folder_name))
    mkdir(data_folder_name)
end

figs_folder_name = 'final_figs';
if not(isfolder(figs_folder_name))
    mkdir(figs_folder_name)
end

%% ---- steady-state error probability
% -- Figs 4 and 5
model=load('A_Laplace.mat');
A_N=model.A_N;
T=3000;Tm=5000;m=100;Ns=200;

seed=0; 
delta=1./(10:10:100);
[Error,Error_ave,Error_steady,Error_steady_ave] = Laplace_MC(seed,model,delta,T,Tm,m,Ns);
filename = strcat('Laplace_steady_seed=', num2str(seed), '.mat');
fullfilename=fullfile(data_folder_name,filename);
save(fullfilename, 'delta', 'T', 'Tm', 'm', 'Ns', 'Error', 'Error_ave', 'Error_steady', 'Error_steady_ave');

plot_steady_network(delta,Error_steady_ave,figs_folder_name)
plot_steady_agent(A_N,delta,Error_steady,figs_folder_name)

% -- Fig. 7
model=load('A_Laplace.mat');
T=3000;Tm=10000;m=100;Ns=500;

seed=1015;
delta=0.01;
[Error,Error_ave,Error_steady,Error_steady_ave] = Laplace_MC(seed,model,delta,T,Tm,m,Ns);
filename = strcat('Laplace_steady_delta=', num2str(delta), '_seed=', num2str(seed), '.mat');
fullfilename=fullfile(data_folder_name,filename);
save(fullfilename, 'delta', 'T', 'Tm', 'm', 'Ns', 'Error', 'Error_ave', 'Error_steady', 'Error_steady_ave');

%% ---- instantaneous error probability
% -- Fig. 6
model=load('A_Laplace.mat');
T=3000;Tm=5000;m=100;Ns=200;

seed=0;
delta=0.01;T_tran=[1000,2000];
[Error,Error_ave] = Laplace_MC_transient(seed,model,delta,T,Tm,Ns,T_tran);
filename = strcat('Laplace_transient_seed=', num2str(seed), '.mat');
fullfilename=fullfile(data_folder_name,filename);
save(fullfilename, 'delta', 'T', 'Tm', 'Ns', 'T_tran', 'Error', 'Error_ave');

plot_transient_network(T,Error_ave,figs_folder_name)


%% ----------------------   Functions   -----------------------
% MC simulations

function [Error,Error_ave,Error_steady,Error_steady_ave] = Laplace_MC(seed,model,delta,T,Tm,m,Ns)
%% stationary environments
rng(seed)
len=length(delta);
N=model.N;A_N=model.A_N;
Error=zeros(A_N*N,T,len);
Error_steady=zeros(A_N*N,len);
for i=1:len
    for j=1:Ns
        error=Laplace_tensor(model,delta(i),T,Tm);
        Error(:,:,i)=Error(:,:,i)+error;
    end
    Error(:,:,i)=Error(:,:,i)/Ns;

    % Calculate the steady-state error probability: the avearge of error values 
    % within the last m time instants:

    Error_steady(:,i)=sum(Error(:,end-m+1:end,i),2)/m;

end

Error_steady_ave=zeros(A_N,len);
Error_ave=zeros(A_N,T,len);
for i=1:len
    for j=1:A_N
        Error_steady_ave(j,i)=sum(Error_steady((j-1)*N+1:j*N,i))/N;
        Error_ave(j,:,i)=sum(Error(1+(j-1)*N:j*N,:,i))/N;
    end
end
end

function [Error,Error_ave] = Laplace_MC_transient(seed,model,delta,T,Tm,Ns,T_tran)
%% non-stationary environments
rng(seed)
len=length(delta);
N=model.N;A_N=model.A_N;
Error=zeros(A_N*N,T,len);
Error_ave=zeros(A_N,T,len);
for i=1:len
    for j=1:Ns
        error=Laplace_tensor_transient(model,delta(i),T,Tm,T_tran);
        Error(:,:,i)=Error(:,:,i)+error;
    end
    Error(:,:,i)=Error(:,:,i)/Ns;

end

for i=1:len
    for j=1:A_N
        Error_ave(j,:,i)=sum(Error(1+(j-1)*N:j*N,:,i))/N;
    end
end
end


% subfunctions for MC simulations

function error=Laplace_tensor(model,delta,T,Tm)
% ---------------------------- Code Descriptions --------------------------
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
% -------------------------------------------------------------------------

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
end

function error=Laplace_tensor_transient(model,delta,T,Tm,T_tran)
% --------------------------  Code Descriptions ---------------------------
% This is the code for running multiple Monte Carlo simulations of the ASL
%  process under non-stationary environments in Fig. 6.

% Input:
% -model: parameters for the social learning task
% -delta: step-size
% -T: terminal time
% -Tm: number of Monte Carlo runs
% -T_tran: time instants of the variation of the true state

% Output:
% -error: instantaneous error probability of each agent for different
%         combination policies, averaged over Tm Monte Carlo runs.
% -------------------------------------------------------------------------

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
Lambda=0*ones(NN*N,Tm,M-1);
error=zeros(NN*N,T);
X=zeros(N,Tm,M-1);
% updating rule
for t=1:T_tran(1)
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
for t=T_tran(1)+1:T_tran(2)
    a=rand(N,Tm)-0.5;
    Data=mu(:,2)-sigma(:,2).*sign(a).*log(1-2*abs(a));
    X(:,:,1)=log(sigma(:,2)./sigma(:,1))+abs(Data-mu(:,2))./sigma(:,2)-abs(Data-mu(:,1))./sigma(:,1);
    X(:,:,2)=log(sigma(:,3)./sigma(:,1))+abs(Data-mu(:,3))./sigma(:,3)-abs(Data-mu(:,1))./sigma(:,1);
    for m=1:M-1
        Lambda(:,:,m)=AA'*((1-delta)*Lambda(:,:,m)+delta*repmat(X(:,:,m),NN,1));
    end
    aux=(Lambda(:,:,1)>=zeros(NN*N,Tm))+(Lambda(:,:,2)<=Lambda(:,:,1));
    error(:,t)=transpose(sum(transpose(aux>0))/Tm);
end
for t=T_tran(2)+1:T
    a=rand(N,Tm)-0.5;
    Data=mu(:,3)-sigma(:,3).*sign(a).*log(1-2*abs(a));
    X(:,:,1)=log(sigma(:,2)./sigma(:,1))+abs(Data-mu(:,2))./sigma(:,2)-abs(Data-mu(:,1))./sigma(:,1);
    X(:,:,2)=log(sigma(:,3)./sigma(:,1))+abs(Data-mu(:,3))./sigma(:,3)-abs(Data-mu(:,1))./sigma(:,1);
    for m=1:M-1
        Lambda(:,:,m)=AA'*((1-delta)*Lambda(:,:,m)+delta*repmat(X(:,:,m),NN,1));
    end
    aux=(Lambda(:,:,2)>=zeros(NN*N,Tm))+(Lambda(:,:,1)<=Lambda(:,:,2));
    error(:,t)=transpose(sum(transpose(aux>0))/Tm);
end
end


% figures

function plot_steady_network(delta,Error_steady_ave,figs_folder_name)
figure;hold on;box on;grid on
set(gcf, 'DefaultTextInterpreter', 'latex')
set(gca,'YScale','log','Fontname','Times New Roman','Fontsize',20);
xlabel('$1/\delta$');ylabel('Error probability');
xlim([1/delta(1),1/delta(end)])
xticks(1./delta)
for i=1:5
    plot(1./delta,Error_steady_ave(i,:),'bo-','Linewidth',1.2,'MarkerSize',7,'MarkerFaceColor','b');
    plot(1./delta,Error_steady_ave(i+5,:),'go-','Linewidth',1.2,'MarkerSize',7,'MarkerFaceColor','g');
end
legend('left-stochastic matrix','doubly-stochastic matrix','Interpreter','latex')
filename=fullfile(figs_folder_name, 'Laplace_steady_network.pdf');
exportgraphics(gcf,filename,'ContentType','vector')
end

function plot_steady_agent(A_N,delta,Error_steady,figs_folder_name)
%% error exponent for 4 agents
km=[1,4,7,10];
figure;
set(gcf, 'DefaultTextInterpreter', 'latex')
for i=1:length(km)
    subplot(2,2,i);hold on;box on;grid on
    set(gca,'Yscale','log','Fontsize',20,'Fontname','Times New Roman');
    xlabel('$1/\delta$');
    title(['Agent ', num2str(km(i))],'FontWeight','normal')
    xlim([1/delta(1),1/delta(end)])
    for j=1:5
        plot(1./delta,Error_steady(km(i)+(j-1)*A_N,:),'b-','Linewidth',1);
        plot(1./delta,Error_steady(km(i)+(j+4)*A_N,:),'g-','Linewidth',1);
    end
end
filename=fullfile(figs_folder_name, 'Laplace_steady_agent.pdf');
exportgraphics(gcf,filename,'ContentType','vector')
end

function plot_transient_network(T,Error_ave,figs_folder_name)
%% Network Transient Error Probability v.s. Time 
gray1=[0.7,0.7,0.7];
gray2=[0.5,0.5,0.5];
gray3=[0.2,0.2,0.2];
figure;hold on;box on;grid on
set(gcf, 'DefaultTextInterpreter', 'latex')
nPos = get(gca, 'Position');
set(gca,'YScale','log','Fontname','Times New Roman','Fontsize',20,'Position',[nPos(1),nPos(2)+nPos(4)/8,nPos(3),nPos(4)-nPos(4)/8]);
xlabel('$i$');ylabel('Error probability');
j=1;
for i=1:5
    h1=plot(1:T,Error_ave(i,:,j),'b', LineWidth=1.2);
    h2=plot(1:T,Error_ave(i+5,:,j),'g',LineWidth=1.2);
end
legend([h1 h2],'left-stochastic matrix','doubly-stochastic matrix','Interpreter','latex')
an1=annotation('textbox',[nPos(1),nPos(2)-nPos(4)/8,nPos(3)/3,nPos(4)/12],'String','$\theta_1$','BackgroundColor',gray1);
an2=annotation('textbox',[nPos(1)+nPos(3)/3,nPos(2)-nPos(4)/8,nPos(3)/3,nPos(4)/12],'String','$\theta_2$','BackgroundColor',gray2);
an3=annotation('textbox',[nPos(1)+2*nPos(3)/3,nPos(2)-nPos(4)/8,nPos(3)/3,nPos(4)/12],'String','$\theta_3$','BackgroundColor',gray3);
set([an1,an2,an3],'Interpreter','latex', 'HorizontalAlignment','center', ...
    'VerticalAlignment','cap','Fontsize',20,'FaceAlpha',0.4,'FitBoxToText','off')
filename = fullfile(figs_folder_name, 'Laplace_transient_network.pdf');
exportgraphics(gcf,filename,'ContentType','vector')
end
