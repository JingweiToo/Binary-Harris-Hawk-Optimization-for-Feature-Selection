%-------------------------------------------------------------------------%
%  Binary Harris Hawk Optimization (BHHO) source codes demo version       %
%                                                                         %
%  Programmer: Jingwei Too                                                %
%                                                                         %
%  E-Mail: jamesjames868@gmail.com                                        %
%-------------------------------------------------------------------------%


%---Inputs-----------------------------------------------------------------
% feat:   features
% label:  labelling
% N:      Number of hawks
% T:      Maximum number of iterations
%---Outputs----------------------------------------------------------------
% sFeat:  Selected features
% Sf:     Selected feature index
% Nf:     Number of selected features
% curve:  Convergence curve
%--------------------------------------------------------------------------



%% Binary Harris Hawk Optimization
clc, clear, close; 
% Benchmark data set 
load ionosphere.mat; 
% Set 20% data as validation set
ho=0.2; 
% Hold-out method
HO=cvpartition(label,'HoldOut',ho,'Stratify',false);
% Parameter setting
N=10; T=100;
% Binary Harris Hawk Optimization
[sFeat,Sf,Nf,curve]=jBHHO(feat,label,N,T,HO);
% Plot convergence curve
figure(); plot(1:T,curve); xlabel('Number of iterations');
ylabel('Fitness Value'); title('BHHO'); grid on;




