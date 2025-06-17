ARMA(1,1) is a combination of AR(1) ie autocorrection model and MA(1) ie moving average.

intuitivly lets create an AR model first.
AR:= a model that aims to express a time series as a function of its past values; Xt = β0 + β1 Xt-1 + β2 Xt-2 +...+ βp Xt-p +ℇt, ℇt ~ N(0, σ^2) as is a white noise.
Stationarity:= a process is weakly stationary iff; E(X)=µ, Var(X)=σ^2, Cov(Xt, Xt-h) is dependent on h not t.
For AR(1) stationary iff |β|<1, for AR(p) stationary iff β(z)= 1 - β1 z - β2 z^2 - β3 z^3 ... =0 are not in unit circle. *1

estimating the coefficients of β we have a few techniques; 

1) OLS, we can write it into matrix for Y = Xβ +ℇ (Y=Xt, X is subsequent Xs).
OLS estimator := minβ ℇhat'ℇhat =(Y-Xβ)'(Y-Xβ) = Y'Y -Y'Xβ -β'X'Y +β'X'Xβ due to it being a symetric matrix = Y'Y -2β'X'Y +β'X'Xβ
FOC: ∂/∂β = 0 = -2X'Y +2X'Xβ
βhat = (X'X)^-1 X'Y lets label this (ARE 1) for when we can compare
pros; doesnt require stationarity, cons; No autocorrection optimality, endogeniety breaks model.

2) MLE, yet another method I didnt need to reseach yay!
for large p which your likely using this will blow up your computor, and with assumed gaussian errors its the same as OLS so will just provide the math.
ℇ~N(0,σ^2), => L(β,σ^2) = SUM from t=p+1 to T of (2pi σ^2)^-1/2 exp{-(Xt - SUMj=1 to p βj Xt-j)^2 / 2σ^2 }
l(β,σ^2) = P-T /2 - SUM t=p+1 to T {Xt - SUMj=1 to p βj Xt-j)^2 / 2σ^2 } 
use computation techniques to minimise this value, this section was more of a fun fact bit ony 1,3 will be used.

3)Yule Walker, need to assume stationary, let ɣ(h)= Cov(Xt, Xt-h), no β0
E{ Xt Xt-k } = E{ β1 Xt-1 Xt-k + β2 Xt-2 Xt-k +...+ βp Xt-p Xt-k + ℇt Xt-k} = β1 E(Xt-k Xt-1) +...+βp E(Xt-p Xt-k), E(ℇt Xt-k)=0 as white noise uncorrelated to past values
=> ɣ(k)= β1 ɣ(k-1) + β2 ɣ(k-2) +...+ βp ɣ(k-p)
let β = (β1, β2,..., βp)' , 
  ɣp = (ɣ(1), ɣ(2),..., ɣ(p))', 
  Γp = matrix | ɣ(0), ɣ(1),..., ɣ(p-1) |
              | ɣ(1), ɣ(0),..., ɣ(p-2) |
              | ɣ(2), ɣ(1),..., ɣ(p-3) |... you've seen a covarience matrix before.
  then Γp β = ɣp or βhat = Γp^-1 ɣp

  now in practise we dont know the covariances so we have to guess them:
  ɣhat(h) = 1/T Sum t=h+1 to T of(Xt - Xbar)(Xt-n - Xbar) => Γphat βhat = ɣphat

  Wow youre still here, lets acc code one up now; we will be using yfinance data on s&p AR(p) with closing price:
############################################################################################################################################
#AR(p)

import yfinance as yf #packages
import numpy as np

# Parameters
T = 200 #how many past data points you will use 
p = 5 #amount of beta terms 

# Get data
data = yf.download('^GSPC', start='2020-01-01', end='2025-06-15', interval='1d')
X = data['Close'].tail(T).values  # convert to numpy array

def sam_AC(X, h):  # Sample autocovariance function
    T = len(X)
    Xbar = np.mean(X)
    cov = 0
    for t in range(h, T):
        cov += (X[t] - Xbar) * (X[t - h] - Xbar)
    return cov / T

def toeplitz(c):
    n = len(c)
    TP = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            TP[i, j] = c[abs(i - j)].item()
    return TP

def YW(X, p):  # Yule-Walker Estimation
    gamma = np.array([sam_AC(X, h) for h in range(p+1)])
    Gamma_p = toeplitz(gamma[:-1])
    gamma_p = gamma[1:]
    beta_hat = np.linalg.solve(Gamma_p, gamma_p)
    
    # Ensure both are 1D arrays before the dot product
    beta_hat = beta_hat.flatten()
    gamma_p = gamma_p.flatten()
    
    sigma2 = gamma[0] - np.dot(beta_hat, gamma_p)
    return beta_hat, sigma2

# Run estimation
beta_hat, sigma2 = YW(X, p)
print("Estimated AR coefficients (beta_hat):", beta_hat)
print("Estimated noise variance (sigma^2):", sigma2)

#check for stationarity 

poly_coefs = np.concatenate(([1], -beta_hat))

# Find roots
roots = np.roots(poly_coefs)

print("Roots of the characteristic polynomial:", roots)
print("Absolute values of roots:", np.abs(roots))

# Stationarity check
if np.all(np.abs(roots) > 1):
    print("The AR process is stationary")
else:
    print("The AR process is NOT stationary")
############################################################################################################################################
*1 going back to why if its in the roots are in the unit circle its not stationary just because its interesting:
lets add a lag opperator L such that LXt = Xt-1, L Xt-1 = Xt-2 or L^2 = Xt-2
then Xt = β1 Xt-1 + β2 Xt-2 +...+ βp Xt-p +ℇt becomes
Xt = Xt{ β1 L + β2 L^2 +...+ L^p} +ℇt, or moving to one side
ℇt = Xt{ 1 - β1 L - β2 L^2 -...- L^p}, compact this
ℇt = β(L) Xt => Xt = β(L)^-1 ℇt, which is an infinite sum that will only converge if past shocks have diminishing effect which wont happen with roots within the unit circle.
  

