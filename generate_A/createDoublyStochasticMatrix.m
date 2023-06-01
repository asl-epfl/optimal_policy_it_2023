%% ------------ Code Descriptions ----------------
% This is the code for generating doubly-stochastic matrices.

function A_double=createDoublyStochasticMatrix(N,A,SymFlag)
A_double=rand(N).*A;
A_double=A_double./sum(A_double);
flag=0;
N_iter=10000;
k=0;
while (flag==0)&&(k<N_iter)
    for i=1:N % row normalization
        A_double(i,:)=A_double(i,:)/sum(A_double(i,:));
    end
    for i=1:N % column normalization
        A_double(:,i)=A_double(:,i)/sum(A_double(:,i));
    end
    if SymFlag==1
        A_double=0.5*(A_double+A_double');
    end
    if isequal(sum(A_double),ones(N,1))&&isequal(sum(A_double'),ones(N,1))
       flag=1;
    end
    k=k+1;
end
    