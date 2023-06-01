% --------------------------- Code Descriptions ---------------------------
% This is the code for Fig. 2: LMGFs.
% The LMGF expression is: 
%       { -t * (0.01 - mu^2) + t^2 * (0.01 - mu)^2 } / 2
% -------------------------------------------------------------------------

M=100;m=4;
x=[];y=[];z=[];xyz=zeros(length(-m:1/M:m),1);
for i=-m:1/M:m
    x=[x,0.02*i^2];
    y=[y,(0.03*i+0.01*i^2)/2];
    z=[z,0.01*(i^2-i)/2];
end

figure;hold on;box on
set(gcf,'DefaultTextInterpreter','latex');
set(gca,'Fontname','Times New Roman','Fontsize',20,'YAxisLocation','origin','XAxisLocation','origin');
xlabel('$t$');
xlim([-m,m]);
xticks(-m:m)
ylim([-0.02,0.04])
plot(-m:1/M:m,x,'-','linewidth',2);
plot(-m:1/M:m,y,'-','linewidth',2);
plot(-m:1/M:m,z,'-','linewidth',2);
plot(-m:1/M:m,xyz,'-','linewidth',2);
legend('$\Lambda_1(t;\theta_2)$','$\Lambda_2(t;\theta_2)$','$\Lambda_3(t;\theta_2)$','$\Lambda_4(t;\theta_2)$', ...
    'Interpreter','latex','NumColumns',2,'Location','best','Box','off')
exportgraphics(gcf,'final_figs/LMGF.pdf','ContentType','vector')