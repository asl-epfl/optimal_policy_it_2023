% ------------ Code Descriptions ------------
% This is the main function for generating figures in Section V.B.

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
% -- Figs. 10 and 11
T=3000;Tm=10000;m=100;Ns=100;
seed=0;
model = load('A_noisy_Gaussian.mat');
N=model.N; A_N = size(model.A,3);
delta=1./(10:20:200);
[Error,Error_ave,Error_steady,Error_steady_ave] = Gaussian_MC(seed,model,delta,T,Tm,m,Ns);
filename = strcat('Gaussian_steady_seed=', num2str(seed), '.mat');
fullfilename=fullfile(data_folder_name,filename);
save(fullfilename, 'delta', 'T', 'Tm', 'm', 'Ns', 'Error', 'Error_ave', 'Error_steady', 'Error_steady_ave');

plot_steady_network(delta, Error_steady_ave,figs_folder_name)
plot_steady_agent(N,A_N,delta,Error_steady,figs_folder_name)

% -- Fig. 12 
model = load('A_noisy_Gaussian.mat');
T=3000;Tm=10000;m=100;Ns=500;
seed=0;
delta=0.01;
[Error,Error_ave,Error_steady,Error_steady_ave] = Gaussian_MC(seed,model,delta,T,Tm,m,Ns);
filename = strcat('Gaussian_steady_delta=', num2str(delta), '_seed=', num2str(seed), '.mat');
fullfilename=fullfile(data_folder_name,filename);
save(fullfilename, 'delta', 'T', 'Tm', 'm', 'Ns', 'Error', 'Error_ave', 'Error_steady', 'Error_steady_ave');

%% ---- instantaneous error probability
%-- Fig. 13
model = load('A_noisy_Gaussian.mat');
T=3000;Tm=5000;Ns=200;
delta=0.01;T_tran=[1000;2000];
seed=0;
[Error,Error_ave]= Gaussian_MC_transient(seed,model,delta,T,Tm,Ns,T_tran);
filename = strcat('Gaussian_transient_delta=', num2str(delta), '_seed=', num2str(seed), '.mat');
fullfilename=fullfile(data_folder_name,filename);
save(fullfilename, 'delta', 'T', 'Tm', 'Ns', 'T_tran', 'Error', 'Error_ave');

plot_transient_network(T,Error_ave,figs_folder_name)

% -- Fig. 14
model = load('A_noisy_Gaussian_changing.mat');
T=3000;Tm=5000;Ns=200;
delta=0.01;T_tran=[1000;2000];
seed=0;
[Error,Error_ave]= Gaussian_MC_transient(seed,model,delta,T,Tm,Ns,T_tran);
filename = strcat('Gaussian_transient_changing_delta=', num2str(delta), '_seed=', num2str(seed), '.mat');
fullfilename=fullfile(data_folder_name,filename);
save(fullfilename, 'delta', 'T', 'Tm', 'Ns', 'T_tran', 'Error', 'Error_ave');

plot_transient_network_changing(T,Error_ave,figs_folder_name)


%% --------------------- functions ----------------------
% MC simulations

function [Error,Error_ave,Error_steady,Error_steady_ave]= Gaussian_MC(seed,model,delta,T,Tm,m,Ns)
%% stationary environments
rng(seed);
len=length(delta);
N=model.N;A_N=size(model.A,3);
Error=zeros(A_N*N,T,len);
Error_steady=zeros(A_N*N,len);
for i=1:len
    for j=1:Ns
        error=Gaussian_tensor(model,delta(i),T,Tm);
        Error(:,:,i)=Error(:,:,i)+error;
    end
    Error(:,:,i)=Error(:,:,i)/Ns;
    Error_steady(:,i)=sum(Error(:,end-m+1:end,i),2)/m;
end

Error_steady_ave=zeros(A_N,len);
Error_ave=zeros(A_N,T,len);

for i=1:len
    for j=1:A_N
        Error_steady_ave(j,i)=sum(Error_steady((j-1)*N+1:j*N,i))/N;
        Error_ave(j,:,i)=sum(Error(1+(j-1)*N:j*N,:,i),1)/N;
    end
end

end

function [Error,Error_ave]= Gaussian_MC_transient(seed,model,delta,T,Tm,Ns,T_tran)
%% non-stationary environments
rng(seed);
len=length(delta);
N=model.N;A_N=size(model.A,3);
Error=zeros(A_N*N,T,len);
Error_ave=zeros(A_N,T,len);

for i=1:len
    for j=1:Ns
        error=Gaussian_tensor_transient_changing(model,delta,T,Tm,T_tran);
        Error(:,:,i)=Error(:,:,i)+error;
    end
    Error(:,:,i)=Error(:,:,i)/Ns;
end

for i=1:len
    for j=1:A_N
        Error_ave(j,:,i)=sum(Error(1+(j-1)*N:j*N,:,i),1)/N;
    end
end

end


% figures

function plot_steady_network(delta, Error_steady_ave, figs_folder_name)
%% Network Steady-state Error Probability v.s. Step-size
figure;hold on;box on;grid on
set(gcf, 'DefaultTextInterpreter', 'latex')
set(gca,'YScale','log','Fontname','Times New Roman','Fontsize',20);
xlabel('$1/\delta$');
ylabel('Error probability');
xlim([1/delta(1), 1/delta(end)]);
for i=1:4
    plot(1./delta,Error_steady_ave(i,:),'bo-','Linewidth',1.2,'MarkerSize',7,'MarkerFaceColor','b');
    plot(1./delta,Error_steady_ave(i+5,:),'go-','Linewidth',1.2,'MarkerSize',7,'MarkerFaceColor','g');
end
h1=plot(1./delta,Error_steady_ave(5,:),'bo-','Linewidth',1.2,'MarkerSize',7,'MarkerFaceColor','b');
h2=plot(1./delta,Error_steady_ave(10,:),'go-','Linewidth',1.2,'MarkerSize',7,'MarkerFaceColor','g');
h3=plot(1./delta,Error_steady_ave(end,:),'ro-','Linewidth',1.2,'MarkerSize',7,'MarkerFaceColor','r');
legend([h1 h2 h3],'left-stochastic matrix','doubly-stochastic matrix','optimal combination policy','Interpreter','latex')
filename = fullfile(figs_folder_name, 'Gaussian_steady_network.pdf');
exportgraphics(gcf,filename,'ContentType','vector')
end

function plot_steady_agent(N,A_N,delta,Error_steady,figs_folder_name)
%% Node Steady-state Error Probability v.s. Step-size
% Select agent 1,4,7,10 as examples
km=[1,4,7,10];
figure;
set(gcf, 'DefaultTextInterpreter', 'latex')
for i=1:4
    subplot(2,2,i);hold on;box on;grid on
    set(gca,'Yscale','log','Fontsize',20,'Fontname','Times New Roman');
    xlabel('$1/\delta$');
    title(['Agent ', num2str(km(i))],'FontWeight','normal')
    xlim([1/delta(1),1/delta(end)])
    for j=1:5
        plot(1./delta,Error_steady(km(i)+(j-1)*(A_N-1),:),'b-','Linewidth',1.2);
        plot(1./delta,Error_steady(km(i)+(j+5)*(A_N-1),:),'g-','Linewidth',1.2);
    end
    plot(1./delta,Error_steady(km(i)+N*(A_N-1),:),'r-','Linewidth',1.2);
end
filename = fullfile(figs_folder_name, 'Gaussian_steady_agent.pdf');
exportgraphics(gcf,filename,'ContentType','vector')
end

function plot_transient_network(T,Error_ave,figs_folder_name)
%% Network Transient Error Probability v.s. Time
gray1=[0.7,0.7,0.7];
gray2=[0.5,0.5,0.5];
gray3=[0.3,0.3,0.3];
figure;hold on;box on;grid on
set(gcf, 'DefaultTextInterpreter', 'latex')
nPos = get(gca, 'Position');
set(gca,'YScale','log','Fontname','Times New Roman','Fontsize',20,'Position',[nPos(1),nPos(2)+nPos(4)/8,nPos(3),nPos(4)-nPos(4)/8]);
xlabel('$i$','Interpreter','latex');
ylabel('Error probability');
j=1;
for i=1:5
    h1=plot(1:T,Error_ave(i,:,j),'b','LineWidth',1.2);
    h2=plot(1:T,Error_ave(i+5,:,j),'g','LineWidth',1.2);
end
h3=plot(1:T,Error_ave(end,:,j),'r','LineWidth',1.2);
legend([h1 h2 h3],'left-stochastic matrix','doubly-stochastic matrix','optimal combination matrix','Interpreter','latex')
an1=annotation('textbox',[nPos(1),nPos(2)-nPos(4)/8,nPos(3)/3,nPos(4)/12],'String','$\theta_1$','BackgroundColor',gray1);
an2=annotation('textbox',[nPos(1)+nPos(3)/3,nPos(2)-nPos(4)/8,nPos(3)/3,nPos(4)/12],'String','$\theta_2$','BackgroundColor',gray2);
an3=annotation('textbox',[nPos(1)+2*nPos(3)/3,nPos(2)-nPos(4)/8,nPos(3)/3,nPos(4)/12],'String','$\theta_3$','BackgroundColor',gray3);
set([an1,an2,an3],'Interpreter','latex','HorizontalAlignment','center', ...
    'VerticalAlignment','cap','Fontsize',20,'FaceAlpha',0.4,'FitBoxToText','off')
filename = fullfile(figs_folder_name, 'Gaussian_transient_network.pdf');
exportgraphics(gcf,filename,'ContentType','vector')
end

function plot_transient_network_changing(T,Error_ave,figs_folder_name)
gray1=[0.7,0.7,0.7];
gray2=[0.5,0.5,0.5];
gray3=[0.3,0.3,0.3];
figure;hold on;box on;grid on
set(gcf, 'DefaultTextInterpreter', 'latex')
nPos = get(gca, 'Position');
set(gca,'YScale','log','Fontname','Times New Roman','Fontsize',20,'Position',[nPos(1),nPos(2)+nPos(4)/8,nPos(3),nPos(4)-nPos(4)/8]);
xlabel('$i$','Interpreter','latex');ylabel('Error probability');
h3=plot(1:T,Error_ave(11,:),'b-','Linewidth',1.2);
h4=plot(1:T,Error_ave(12,:),'g-','Linewidth',1.2);
h5=plot(1:T,Error_ave(13,:),'r-','Linewidth',1.2);
legend([h3 h4 h5],'$A^\star_1$','$A^\star_2$','$A^\star_3$','Interpreter','latex')
an1=annotation('textbox',[nPos(1),nPos(2)-nPos(4)/8,nPos(3)/3,nPos(4)/12],'String','$\theta_1$','BackgroundColor',gray1);
an2=annotation('textbox',[nPos(1)+nPos(3)/3,nPos(2)-nPos(4)/8,nPos(3)/3,nPos(4)/12],'String','$\theta_2$','BackgroundColor',gray2);
an3=annotation('textbox',[nPos(1)+2*nPos(3)/3,nPos(2)-nPos(4)/8,nPos(3)/3,nPos(4)/12],'String','$\theta_3$','BackgroundColor',gray3);
set([an1,an2,an3],'Interpreter','latex','HorizontalAlignment','center', ...
    'VerticalAlignment','cap','Fontsize',20,'FaceAlpha',0.4,'FitBoxToText','off')
filename = fullfile(figs_folder_name, 'Gaussian_transient_network_changing.pdf');
exportgraphics(gcf,filename,'ContentType','vector')
end

