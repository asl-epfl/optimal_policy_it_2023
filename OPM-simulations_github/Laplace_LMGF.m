%% ------------ Code Descriptions ----------------
% This is the code for calculating the average LMGF
% $\Lambda_{ave}(t;\pi,\theta)$ for the learning task in Section V.A.

% This function will be used in Laplace_adaptation_time.m for obtaining 
% (i)   the critical value $t^\star(\theta)$ 
% (ii)  the \theta-related error exponent $\Phi(\pi,\theta)$
% (iii) the approximated adaptation time $T_{ASL}(\pi,\omega)$

% Input:
% -N:   parameters for the social learning task
% -mu:  vector of the mean values of L_k(|\theta), i.e., the wrong hypothesis.
% -mu1: mean value of L_k(|\theta_1), i.e., the true hypothesis.
% -t:   time variable in LMGF
% -P:   Perron eigenvector

% Output:
% -LMGF: vector of the value of $\Lambda_{ave}(t;\pi,\theta)$ at t, i.e.,
%        $log E[e^{tx_{ave,i}(\pi,\theta)}$ for the wrong hypotheses.


function LAMBDA=Laplace_LMGF(N,mu,mu1,t,P)
LAMBDA=0;
for i=1:N
    DD=mu(i)-mu1;
    if P(i)*t+0.5~=0
        if DD>0
            LAMBDA=LAMBDA+log(0.5*exp(-DD*(P(i)*t+1))+0.5*exp(DD*P(i)*t)+0.5*exp(-0.5*DD)*sinh(DD*(P(i)*t+0.5))/(P(i)*t+0.5));
        else
            LAMBDA=LAMBDA+log(0.5*exp(DD*(P(i)*t+1))+0.5*exp(-DD*P(i)*t)-0.5*exp(0.5*DD)*sinh(DD*(P(i)*t+0.5))/(P(i)*t+0.5));
        end
    else
        if DD>0
            LAMBDA=LAMBDA+log(0.5*exp(-DD*(P(i)*t+1))+0.5*exp(DD*P(i)*t)+0.5*exp(-0.5*DD)*DD);
        else
            LAMBDA=LAMBDA+log(0.5*exp(DD*(P(i)*t+1))+0.5*exp(-DD*P(i)*t)-0.5*exp(0.5*DD)*DD);
        end  
    end
end
end
    