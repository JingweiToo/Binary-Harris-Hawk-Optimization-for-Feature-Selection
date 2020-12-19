# Binary Harris Hawk Optimization for Feature Selection

![Wheel](https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/7cf53ce5-1104-490a-97a7-b86aa2b1c17d/f5d1aef5-e893-4569-92ca-b269d967dd77/images/1595488674.JPG)


## Introduction
* This toolbox offers Binary Harris Hawk Optimization ( BHHO )  
* The < Main.m file > illustrates the example of how BHHO can solve the feature selection problem using benchmark data-set. 

## Input
* *feat*     : feature vector ( Instances *x* Features )
* *label*    : label vector ( Instances *x* 1 )
* *N*        : number of hawks
* *max_Iter* : maximum number of iterations


## Output
* *sFeat*    : selected features
* *Sf*       : selected feature index
* *Nf*       : number of selected features
* *curve*    : convergence curve


### Example
```code
% Benchmark data set 
load ionosphere.mat; 

% Set 20% data as validation set
ho = 0.2; 
% Hold-out method
HO = cvpartition(label,'HoldOut',ho);

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

```

## Requirement
* MATLAB 2014 or above
* Statistics and Machine Learning Toolbox


## Cite As
```code
@article{too2019new,
  title={A new quadratic binary harris hawk optimization for feature selection},
  author={Too, Jingwei and Abdullah, Abdul Rahim and Mohd Saad, Norhashimah},
  journal={Electronics},
  volume={8},
  number={10},
  pages={1130},
  year={2019},
  publisher={Multidisciplinary Digital Publishing Institute}
}

```


