function [sFeat,Sf,Nf,curve] = jBHHO(feat,label,N,max_Iter,HO)

beta = 1.5;
ub   = 1; 
lb   = 0;

fun = @jFitnessFunction; 
dim = size(feat,2);
X   = zeros(N,dim); 
for i = 1:N
  for d = 1:dim
    if rand() > 0.5
      X(i,d) = 1;
    end
  end
end
fitR = inf; 
fit  = zeros(1,N); 
Y    = zeros(1,dim); 
Z    = zeros(1,dim);

curve = inf;  
t = 1; 
%---Iteration start-------------------------------------------------
while t <= max_Iter
  for i = 1:N
    fit(i) = fun(feat,label,X(i,:),HO);
    if fit(i) < fitR
      fitR = fit(i);
      Xrb  = X(i,:);
    end
  end
  X_mu = mean(X,1);
  for i = 1:N
    E0 = -1 + 2 * rand();
    E  = 2 * E0 * (1 - (t / max_Iter)); 
    if abs(E) >= 1
      q = rand(); 
      if q >= 0.5
        k  = randi([1,N]);
        r1 = rand();
        r2 = rand();
        for d = 1:dim
          Xn = X(k,d) - r1 * abs(X(k,d) - 2 * r2 * X(i,d));
          S  = 1 / (1 + exp(-Xn));
          if rand() < S
            X(i,d)= 1;
          else 
            X(i,d) = 0;
          end
        end
      elseif q < 0.5
        r3 = rand();
        r4 = rand();
        for d = 1:dim
          Xn = (Xrb(d) - X_mu(d)) - r3 * (lb + r4 * (ub - lb));
          S  = 1 / (1 + exp(-Xn));
          if rand() < S
            X(i,d) = 1;
          else
            X(i,d) = 0;
          end
        end
      end
    elseif abs(E) < 1
      J = 2 * (1 - rand());
      r = rand();
      if r >= 0.5  &&  abs(E) >= 0.5
        for d = 1:dim
          DX = Xrb(d) - X(i,d);
          Xn = DX - E * abs(J * Xrb(d) - X(i,d));
          S  = 1 / (1 + exp(-Xn));
          if rand() < S
            X(i,d) = 1;
          else
            X(i,d) = 0;
          end
        end
      elseif r >= 0.5  &&  abs(E) < 0.5
        for d = 1:dim
          DX = Xrb(d) - X(i,d);
          Xn = Xrb(d) - E * abs(DX);
          S  = 1 / (1 + exp(-Xn));
          if rand() < S
            X(i,d) = 1;
          else
            X(i,d) = 0;
          end
        end
      elseif r < 0.5  &&  abs(E) >= 0.5
        LF = jLevyDistribution(beta,dim); 
        for d = 1:dim
          Yn = Xrb(d) - E * abs(J * Xrb(d) - X(i,d));
          S  = 1 / (1 + exp(-Yn));
          if rand() < S
            Y(d) = 1;
          else
            Y(d) = 0;
          end
          Zn = Y(d) + rand() * LF(d);
          S  = 1 / (1 + exp(-Zn));
          if rand() < S
            Z(d) = 1;
          else
            Z(d) = 0;
          end
        end
        fitY = fun(feat,label,Y,HO);
        fitZ = fun(feat,label,Z,HO);
        if fitY <= fit(i)
          fit(i) = fitY; 
          X(i,:) = Y;
        end
        if fitZ <= fit(i)
          fit(i) = fitZ;
          X(i,:) = Z;
        end
      elseif r < 0.5  &&  abs(E) < 0.5
        LF = jLevyDistribution(beta,dim); 
        for d = 1:dim
          Yn = Xrb(d) - E * abs(J * Xrb(d) - X_mu(d));
          S  = 1 / (1 + exp(-Yn));
          if rand() < S
            Y(d) = 1;
          else
            Y(d) = 0;
          end
          Zn = Y(d) + rand() * LF(d);
          S  = 1 / (1 + exp(-Zn));
          if rand() < S
            Z(d) = 1;
          else
            Z(d) = 0;
          end
        end
        fitY = fun(feat,label,Y,HO); 
        fitZ = fun(feat,label,Z,HO);
        if fitY <= fit(i)
          fit(i) = fitY; 
          X(i,:) = Y;
        end
        if fitZ <= fit(i)
          fit(i) = fitZ; 
          X(i,:) = Z;
        end        
      end
    end
  end
  curve(t) = fitR; 
  fprintf('\nIteration %d Best (BHHO)= %f',t,curve(t))
  t = t + 1;
end
Pos   = 1:dim;
Sf    = Pos(Xrb == 1);
Nf    = length(Sf);
sFeat = feat(:,Sf); 
end


function LF = jLevyDistribution(beta,dim)
nume  = gamma(1 + beta) * sin(pi * beta / 2);
deno  = gamma((1 + beta) / 2) * beta * 2 ^ ((beta - 1) / 2);
sigma = (nume / deno) ^ (1 / beta); 
u     = randn(1,dim) * sigma; 
v     = randn(1,dim);
step  = u ./ abs(v) .^ (1 / beta); 
LF    = 0.01 * step;
end


