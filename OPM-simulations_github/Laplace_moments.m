%% ------------ Code Descriptions ----------------
% This is the code for calculating the first two moments of the individual
% log-likelihood ratio variable $x_{k,i}(\theta)$ for the learning task
% in Section V.A.

% This function will be used in Laplace_adaptation_time. m for calculating
% the mean and variance values (i.e., $m_{ave}(\pi,\theta)$ and 
% $c_{ave}(\pi,\theta)$) associated with the network average variable 
% $x_{ave,i}(\pi,\theta)$, which will then be utilized for obtaining the
% approxiamted adaptation time $T_{ASL}(\pi,\omega)$.

% Input:
% -mu1:    mean value of L_k(|\theta_1), i.e., the true hypothesis
% -mu:     mean value of L_k(|\theta), i.e., the wrong hypothesis.
% -sigma1: variance value of L_k(|\theta_1), i.e., the true hypothesis.
% -sigma:  variance value of L_k(|\theta), i.e., the wrong hypothesis.

% Output:
% -dl:  mean value of variable $x_{k,i}(\theta)$
% -var: variance value of variable $x_{k,i}(\theta)$

function [dl,var]=Laplace_moments(mu1,mu,sigma1,sigma)
if mu1<mu
    b1=log(sigma/sigma1)+(sigma1*mu-sigma*mu1)/(sigma1*sigma);
    a1=(sigma-sigma1)/(sigma1*sigma);
    c1=0.5*(a1*mu1-a1*sigma1+b1);
    b2=log(sigma/sigma1)+(sigma*mu1+sigma1*mu)/(sigma1*sigma);
    a2=-(sigma1+sigma)/(sigma1*sigma);
    c2=0.5*(b2+a2*(mu1+sigma1))-0.5*exp((mu1-mu)/sigma1)*(b2+a2*(mu+sigma1));
    b3=log(sigma/sigma1)+(sigma*mu1-sigma1*mu)/(sigma1*sigma);
    a3=(sigma1-sigma)/(sigma1*sigma);
    c3=0.5*exp((mu1-mu)/sigma1)*(a3*mu+a3*sigma1+b3);
    dl=c1+c2+c3;
    c11=0.5*(a1*mu1-a1*sigma1+b1-dl);
    c22=0.5*(b2-dl+a2*(mu1+sigma1))-0.5*exp((mu1-mu)/sigma1)*(b2-dl+a2*(mu+sigma1));
    c33=0.5*exp((mu1-mu)/sigma1)*(a3*mu+a3*sigma1+b3-dl);
    d1=0.5*(a1*mu1+b1-dl)^2-a1*2*sigma1*c11;
    d2=0.5*(b2+a2*mu1-dl)^2-0.5*(b2+a2*mu-dl)^2*exp((mu1-mu)/sigma1)+a2*2*sigma1*c22;
    d3=0.5*(a3*mu+b3-dl)^2*exp((mu1-mu)/sigma1)+a3*2*sigma1*c33;
    var=d1+d2+d3;
else
    a1=(sigma-sigma1)/(sigma1*sigma);
    b1=log(sigma/sigma1)+(sigma1*mu-sigma*mu1)/(sigma1*sigma);
    c1=0.5*exp((mu-mu1)/sigma1)*(a1*(mu-sigma1)+b1);
    a2=(sigma1+sigma)/(sigma1*sigma);
    b2=log(sigma/sigma1)-(sigma*mu1+sigma1*mu)/(sigma1*sigma);
    c2=0.5*(b2+a2*(mu1-sigma1))-0.5*exp((mu-mu1)/sigma1)*(a2*(mu-sigma1)+b2);
    a3=(sigma1-sigma)/(sigma1*sigma);
    b3=log(sigma/sigma1)+(sigma*mu1-sigma1*mu)/(sigma1*sigma);
    c3=0.5*(a3*(mu1+sigma1)+b3);
    dl=c1+c2+c3;
    c11=0.5*exp((mu-mu1)/sigma1)*(a1*(mu-sigma1)+b1-dl);
    c22=0.5*(b2-dl+a2*(mu1-sigma1))-0.5*exp((mu-mu1)/sigma1)*(a2*(mu-sigma1)+b2-dl);
    c33=0.5*(a3*(mu1+sigma1)+b3-dl);
    d1=0.5*exp((mu-mu1)/sigma1)*(a1*mu+b1-dl)^2-a1*2*sigma1*c11;
    d2=0.5*(a2*mu1+b2-dl)^2-0.5*(a2*mu+b2-dl)^2*exp((mu-mu1)/sigma1)-a2*2*sigma1*c22;
    d3=0.5*(a3*mu1+b3-dl)^2+a3*2*sigma1*c33;
    var=d1+d2+d3;
end
end