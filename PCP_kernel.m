function [P, BS, error]=PCP_kernel(train, ker_para, subsetsize,errorbound)
%% function [P, BS, error]=PCP_kernel(train, ker_para, subsetsize,errorbound)
%Solve the  pivoted Cholesky decomposing kernel matrix with Gaussian
%kernel where the basic set is chosen by pivoting the maximum diag of  
%Schur complement. The whole kernel matrix is approximated by P*P'. 
%% Input:
%  train:             training samples matrix, (one row one sample) 
%  ker_para:      kernel parameters of Gaussian kernel: exp(-ker_para*||x-y||^2).
%  subsetsize:    random subset size upbounds.
%  errorbound:  approximation error bound on trace norm
%% Output: 
%  P, BS:           satisifying K=P*P' and P*P(BS,:)' = K(:,BS);
%  error:            trace(K-P*P').

%% Using the code, please cite the work:
%% "S Zhou.(2016), Sparse LSSVM in primal using Cholesky factorization for large-scale problems.
%%  IEEE Transactions on Neural Networks and Learning Systems, 27(4): 783¨C795"
%

m=size(train,1);   r=1;
BS=zeros(0,subsetsize); NS= 1:m;
P=zeros(m,0);
tr_norm2 = sum(train.*train, 2); %to fast the speed of kernel calculating
%The diag the kernel matrix of gaussian, for other kernel it should be changed
d_K=ones(m,1); 

error(r) = sum(d_K);
%%make the first random selection for gaussian. other kernel will slelect
%%the max of the diagnal.
I=randperm(m);s_in = I(1);
while error(r)>errorbound && r<=subsetsize
    %Next line will change if you use other kernel function
    k_in   = exp(-ker_para*(tr_norm2+tr_norm2(s_in)-train*(2*train(s_in,:)')));
    if r==1
        p=k_in/k_in(s_in); d_K(s_in)=0;
    else
        u=P(s_in,:)';                 nu=sqrt(k_in(s_in)-u'*u);
        p=(k_in-P*u)/nu;         p(BS)=0;d_K(s_in)=0;
    end
    P(:,r)=p; BS(r)=s_in;
    d_K(NS)=d_K(NS) - p(NS).^2; 
    error(r+1) = sum(d_K(NS));
    %find the maximum diag of Shur compliment:
    [~,index]=max(d_K(NS)); s_in=NS(index);NS(index)=[];
    r=r+1;
end
return 

