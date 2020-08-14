function loss = DS_LossPrime(loss_name, param)
%% The structure of directive of the loss;
%% There are about 17 losses, where C means classification and R  means regression.
%% The input loss_name is a string from the following lists:
% {'Chinge-sm','Csquared-hinge','Ctrunc-squared-hinge', 'Ctrunc-least-squares', 
% 'Cramp-sm','Cramp-smp', 'Cnoncon-exp-ext','Cnoncon-log-ext',
% 'Reps-insensitive-sm', 'Rhubber','Rabslute-sm','Rtrunc-abslute-sm', 'Rtrunc-abslute-smp','Rleast-squres' ,
% 'Rtrunc-least-squares','Rnoncon-exp-ext','Rnoncon-log-ext'}
%% The input is a vector in R^6. If given 0, then the default values are used.
 %% The following parameters maybe no use for some losses.
loss.name=loss_name;
 if param(1)>0,     loss.a = param(1);else,        loss.a = 2;     end %truncated parameter
if param(2)>0,      loss.b = param(2);else,        loss.b = 2;     end  %some times used
if param(3)>0,      loss.c = param(3);else,        loss.c = 2;     end %for two new proposed loss only.
if param(4)>0,      loss.p = param(4);else,        loss.p = 10;   end %the parameter to smooth max(x,0) by enptropy function
if param(5)>0,   loss.eps = param(5);else,    loss.eps = 0.1;   end%most used in regression problems
if param(6)>0, loss.delta = param(6);else, loss.delta = 0.1; end%for Hubber loss only.

if lower(loss_name(1))=='c',  loss.type=0; %0--classification; 1---regression; 
elseif lower(loss_name(1))=='r', loss.type =1;
end

switch loss.name
    %case 'Chinge' 
     %   df = @(x)  x>0;
    case 'Chinge-sm' %1
        loss.df = @(x) min(1,exp(x*loss.p))./(1+exp(abs(x)*-loss.p));
        loss.A = loss.p/8;
    case 'Csquared-hinge' 
        loss.df =@(x)  2*max(x,0);
        loss.A = 1;
    case 'Cleast-squres' 
        loss.df = @(x) 2*x;
        loss.A = 1;
    %case 'Cramp'  %not ls-dc loss, forget it
    %    I = r>0; J = r < loss.param;
     %   df(I&J) = 1;
    case 'Ctrunc-squared-hinge' 
        loss.df =@(x) 2*x.*(x>0 & x <sqrt(loss.a));
        loss.A = 1;
    case 'Ctrunc-least-squares'
        loss.df = @(x) 2*x.*(abs(x)<sqrt(loss.a));
        loss.A = 1;
    case 'Cramp-sm'
        loss.df = @(x) 4/loss.a*(max(x, 0).*(x<=loss.a/2) ...
            + max(loss.a - x, 0).*(x>loss.a/2));
        loss.A = 2/loss.a;
    case 'Cramp-smp'
        loss.df =@(x) min(1,exp(loss.p*x))./(1+exp(-loss.p*abs(x))) - ...
            min(1,exp(loss.p*(x-loss.a)))./(1+exp(-loss.p*abs(x-loss.a)));
        loss.A = loss.p/8;
    %case 'Ctrunc-sh-sm'
    %   df = @(x) 2*max(x,0)*(1 - min(1, exp(p*(max(x,0).^2-a)))./(1+exp(-p*abs(max(x,0).^2-a))));
    %case 'Ctrunc-ls-sm' %%will delete 
    %    df = @(x) 2*x.*exp(-loss.p*x.^2)./(exp(-loss.a*loss.p)+exp(-loss.p*x.^2));
    % case 'Cnoncon-exp'
    %    df  = @(x) 2*max(x,0).*exp(max(x,0).^2/-loss.a);
    %case 'Cnoncon-log'
    %   df = @(x) 2*max(x,0)./(1+max(x,0).^2/loss.a);
    case 'Cnoncon-exp-ext'
        loss.df  = @(x) loss.a*loss.c/loss.b*max(x,0).^(loss.c-1).*exp(max(x,0).^loss.c/-loss.b);
        loss.A = dcpara_exp_ext(loss.a, loss.b,loss.c);
    case 'Cnoncon-log-ext'
        loss.df  = @(x) loss.a*loss.c/loss.b*max(x,0).^(loss.c-1)./(1+max(x,0).^loss.c/loss.b);
        loss.A = 1;
    %case 'Reps-insensitive'
     %   df = @(x) -1*(x<-loss.eps) +1*( x>loss.eps);
    case 'Reps-insensitive-sm'
        loss.df =@(x)  - min(1,exp(-loss.p*(loss.eps+x)))./(1+exp(-loss.p*abs(loss.eps+x)))...
            + min(1,exp(-loss.p*(loss.eps-x)))./(1+exp(-loss.p*abs(loss.eps-x)));
        loss.A = loss.p/4;
    %case 'Rabslute'
    %   df = @(x)  1-2*(x<0) ;
    case 'Rhubber'
        loss.df = @(x) -1*(x<-loss.delta) +1*( x>loss.delta) + ...
            x .* (x>=-loss.delta & x<=loss.delta)/loss.delta;
        loss.A = 1/loss.delta;
    case 'Rtrunnc-hubber'
        loss.df = @(x) -1*(x<-loss.delta & x>-loss.a ) +1*( x>loss.delta & x<loss.a) + ...
            x .* (abs(x)<=loss.delta)/loss.delta;
        loss.A = 1/loss.delta;
    case 'Rabslute-sm' % it is case  'Reps-insensitive-sm' with loss.eps=0
        loss.df=@(x)  (min(1,exp(loss.p*x)) - min(1,exp(-loss.p*x)))./(exp(-loss.p*abs(x))+1);
        loss.A = loss.p/4;
    case 'Rtrunc-abslute-sm'
        loss.df = @(x) 4/loss.a * ( x.*(abs(x)<loss.a/2) + ...
            max(loss.a - abs(x) ,0 ).*(abs(x)>loss.a/2) );
        loss.A = 2/loss.a;
    case 'Rtrunc-abslute-smp'
        loss.df =@(x)  min(1,exp(loss.p*x))./(exp(-loss.p*abs(x))+1) - ...
            min(1,exp(loss.p*(x-loss.a))) ./ (exp(-loss.p*abs(x-loss.a)) + 1);
        loss.A = loss.p/4;
    case 'Rleast-squres' 
        loss.df = @(x) 2*x;
        loss.A = 1;
    case 'Rtrunc-least-squares'
        loss.df = @(x) 2*x.*(abs(x)<loss.a^2);
        loss.A = 1;
    case 'Rnoncon-exp-ext'
        loss.df  = @(x) loss.a*loss.c/loss.b* x .^(loss.c-1).*exp( x .^loss.c/-loss.b);
        loss.A = 1;
    case 'Rnoncon-log-ext'
        loss.df  = @(x) loss.a*loss.c/loss.b*abs(x).^(loss.c-1)./(1+abs(x).^loss.c/loss.b);
        loss.A = 1;
    otherwise
        disp('Unkown loss! Please chech again!')
end
return

function A=dcpara_exp_ext(a,b,c)
    hc = (3*(c-1)-sqrt(5*c^2-6*c+1))/(2*c);
    A = a*c/(b^(2/c))*((c-1)*hc^(1-2/c)-c*hc^(2-2/c))*exp(-hc);
    A = A/2;
return