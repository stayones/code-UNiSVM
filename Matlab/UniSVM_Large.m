function [alpha,it] = UniSVM_Large(P, IB, y, lambda, loss, eps0)
%% The code is for the paper "Unified SVM models based LS-DC Losses"(submmitted to "NeurIPS 2020").
% This is the large-scale problem in the paper. The algorithm can solve any kinds of SVM by a unified scheme,     
%   including any SVM models with different losses corresponding to different
%   problems. Given the training data X and a proper kernel function k(x,y), let K =k(X,X').
%   K is factorized as P*P' exactly or approximated by some low-rank approximation methods.
%   
%% Input: 
%   P, IB:      an column full rank matrix P satisfying K=P*P' and K(:,IB)=P*P(IB,:)';
%   y:            the target of preblem;
%   lambda: the regularization parameter;
%   loss:        the chosen LS-DC loss. There are about 20 kinds of losses (See lossfun.m for details) 
%                  loss is a structure including loss.name--str, loss.param-vector, loss.type-'str'('regression' or 'classification'), loss.A-positive sclar(see paper)
%  eps0:       Stop criterion, like 1e-3 or 1e-2.
%% Output:
%  alpha:      output solution of the model. 
%                 The output prediction function is f(x)=sum_i(alpha_i*k(x_{IB(i)}, x)).   
%
%% by sszhou, 2020-5-17.

m = length(y); n = length(IB);
Q = inv(( P' * P + lambda*m/loss.A *eye(n) ) * P(IB,:)'); 
alpha = Q* (P' * y); %start the algorithm from a sparse solution of LSSVM.
v_old = zeros(m,1); it =1;
while ~strcmp(loss.name(2:end), 'least-squres')
    Kx =  P*(P(IB,:)'*alpha); 
    v =  loss_derivative(loss, y, Kx) *  (0.5/loss.A);%disp(norm(v_old - v));
    if norm(v_old - v) < eps0, break; end
    alpha = Q *(P' * (Kx - v) );  v_old = v; it =it+1;
end
return

function df = loss_derivative(loss,y,t)
if       loss.type ==0 %% Classification
    df = - loss.df(1- y.*t).*y;
elseif loss.type == 1%% Regression
    df = - loss.df(y - t);
end