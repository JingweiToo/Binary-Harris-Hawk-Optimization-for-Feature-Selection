%-------------------------------------------------------------------%
%  Binary Harris Hawk Optimization (BHHO) demo version              % 
%-------------------------------------------------------------------%


%---Inputs-----------------------------------------------------------
% feat     : feature vector (instances x features)
% label    : label vector (instance x 1)
% N        : Number of hawks
% max_Iter : Maximum number of iterations

%---Outputs----------------------------------------------------------
% sFeat    : Selected features
% Sf       : Selected feature index
% Nf       : Number of selected features
% curve    : Convergence curve
%--------------------------------------------------------------------


%% Binary Harris Hawk Optimization
clc, clear, close; 
% Benchmark data set 
load ionosphere.mat; 

% Set 20% data as validation set
ho = 0.2; 
% Hold-out method
HO = cvpartition(label,'HoldOut',ho,'Stratify',false);

% Parameter setting
N        = 10; 
max_Iter = 100;
% Binary Harris Hawk Optimization
[sFeat,Sf,Nf,curve] = jBHHO(feat,label,N,max_Iter,HO);

% Plot convergence curve
plot(1:max_Iter,curve);
xlabel('Number of iterations');
ylabel('Fitness Value');
title('BHHO'); grid on;



