% ------------ Code Descriptions ------------
% This is the code for generating the 11 or 13 combination matrices used in
% Section V. B.

% The resulting matrices are included in the model data A_noisy_Gaussian.m 
% and A_noisy_Gaussian_changing.m.

p=0.5;
rng(1);
A0 = rand(N) + eye(N);
A0 = (A0>p);
A0 = A0 + A0'; % symmetric: undirected graph
A0 = sign(A0)./sum(sign(A0)); % averaging rule
A_sign=sign(A0);
A_N=10;
A=zeros(N,N,A_N+1);
A(:,:,1)=A0;

%% generate 5 left-stochastic combination matrices 
for k=1:5
    AA=rand(N).*A_sign;
    AA=AA./sum(AA);
    A(:,:,k+1)=AA;
end

%% generate 5 doubly-stochastic matrices 
for k=1:5
    A(:,:,5+k)=createDoublyStochasticMatrix(N,A_sign,1);
end

Perron_eigenvector=zeros(N,A_N+1);
for i=1:A_N
    [V,D]=eig(A(:,:,i));
    [B,I]=sort(abs(D*ones(N,1)));
    Perron_eigenvector(:,i)=abs(V(:,I(end)))./abs(sum(V(:,I(end))));
end

%% generate the optimal combination matrices 
% A^\star for Figs. 10-13, and A_1^\star, A_2^\star, A_3^\star for Fig. 14.

sigma = [1;1;1;2;2;2;2;3;3;3]; 
noise_level1 = [1;1;1;0.1;0.1;0.1;0.1;0.001;0.001;0.001];
noise_level2 = [0.1;0.1;0.1;0.001;0.001;0.001;0.001;1;1;1];
noise_level3 = [0.001;0.001;0.001;1;1;1;1;0.1;0.1;0.1];

sigma_n1 = sigma .* noise_level1;
sigma_n2 = sigma .* noise_level2;
sigma_n3 = sigma .* noise_level3;
ax=1./(1+sigma_n1./sigma);
Perron_eigenvector(:,A_N+1)=ax./sum(ax);
ax=1./(1+sigma_n2./sigma);
Perron_eigenvector(:,A_N+2)=ax./sum(ax);
ax=1./(1+sigma_n3./sigma);
Perron_eigenvector(:,A_N+3)=ax./sum(ax);
A_opt1=Perron_eigenvector(:,A_N+1).*(A_sign-eye(N));
A_opt2=Perron_eigenvector(:,A_N+2).*(A_sign-eye(N));
A_opt3=Perron_eigenvector(:,A_N+3).*(A_sign-eye(N));
for i=1:N
    A_opt1(i,i)=1-sum(A_opt1(:,i));
    A_opt2(i,i)=1-sum(A_opt2(:,i));
    A_opt3(i,i)=1-sum(A_opt3(:,i));
end
A(:,:,A_N+1)=A_opt1;
A(:,:,A_N+2)=A_opt2;
A(:,:,A_N+3)=A_opt3;
