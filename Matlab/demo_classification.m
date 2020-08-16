
%% demo classification with UniSVMs
%clear
dataset_name='dataset/adult';
Gamma = 2.^-10; Lambda = 10.^-5;IB_max = 1000;
%% Define Loss
params = [0      0      0      0       0              0    ];%using the default settings.
%algs with differnt losses. Read "DS_LossPrime.m" for details.
alg = {'Cleast-squres','Chinge-sm','Csquared-hinge','Ctrunc-squared-hinge','Ctrunc-least-squares',....
    'Cramp-sm','Cramp-smp','Cnoncon-exp-ext','Cnoncon-exp-ext','Cnoncon-exp-ext'};
%% defining loss
loss=cell(1,10); iter=zeros(1,10); time=zeros(1,11); acc=zeros(1,11);
for loss_index=1:length(alg)-2
    loss{loss_index} = DS_LossPrime(alg{loss_index},params);   
end
loss{loss_index+1} = DS_LossPrime('Cnoncon-exp-ext',[2      2      4      0       0              0    ]);
loss{loss_index+2} = DS_LossPrime('Cnoncon-exp-ext',[2      3      4      0       0              0    ]);
eps =1e-2;
load(dataset_name); 
X_te = full(te_instance); y_te = te_label; m = length(tr_label); 
lambda = Lambda;     gamma =Gamma;    IBmax = IB_max;
str =['-q -s 0 -g ' num2str(gamma)  ' -c ' num2str(1/(lambda*m*2))];%for LibSVM
X_tr = full(tr_instance); y_tr = tr_label; 
tic,[P,IB]=PCP_kernel(X_tr,  gamma, IBmax,1e-3*m);t_ker=toc; %obtain K=P*P';
Kt=exp(-gamma*(sum(X_tr(IB,:).* X_tr(IB,:),2) + sum(X_te.*X_te,2)' - 2*X_tr(IB,:)*X_te'));
disp([ dataset_name ':  Training size>>' int2str(m) '   ---Testing size>>' int2str(length(y_te))]); 
for k=1:length(loss)
    tic,   
    [alpha, iter(k)] = UniSVM_Large(P, IB, y_tr, lambda, loss{k}, eps);
    time(k)=toc+t_ker;
    acc(k) = mean(alpha'*Kt.*y_te'>0)*100;
    disp(['UniSVM' num2str(k-1,'%2.0f') '>>'...
        'Test accuracy:' num2str(acc(k),'%2.2f') '  >>time:' num2str(time(k),'%2.2f')...
        '  >>Iter: ' int2str(iter(k)) '  >>lossName: ' loss{k}.name(2:end)]);
end
k=k+1;
disp('Waiting for LibSVM......');
tic, %%libsvm
model = svmtrain_lib(y_tr, X_tr, str );time(k)=toc;
yhat = svmpredict_lib(y_te, X_te,model);
acc(k) = nnz(y_te==yhat)/length(y_te)*100; 
disp(['LibSVM(SVC)>>Test accuracy:' num2str(acc(k),'%2.2f') '>>time:' num2str(time(k),'%2.2f') '>>LibSVM']);
