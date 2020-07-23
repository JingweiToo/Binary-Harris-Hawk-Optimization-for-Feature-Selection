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
% N:      Number of particles
% T:      Maximum number of iterations
% *Note: k-value of KNN & hold-out setting can be modified in jFitnessFunction.m
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
% Parameter setting
N=10; T=100;
% Binary Harris Hawk Optimization
[sFeat,Sf,Nf,curve]=jBHHO(feat,label,N,T);
% Plot convergence curve
figure(); plot(1:T,curve); xlabel('Number of iterations');
ylabel('Fitness Value'); title('BHHO'); grid on;




