import numpy as np

# Q1 (a) (I)
class LinearRegressionFn(): # Benchmark A: Linear Regression Quadratic Loss
    def __init__(self, m=1000, seed=0, noise_std=0.1):
        np.random.seed(seed)
        self.m = m
        self.theta_star = np.array([3.0, 4.0])
        self.X = np.random.randn(m, 2)
        noise = noise_std * np.random.randn(m)
        self.y = self.X @ self.theta_star + noise

    def f(self, theta):
        err = self.X @ theta - self.y
        return 0.5 * np.mean(err**2)

    def df(self, theta):
        err = self.X @ theta - self.y
        return (self.X.T @ err) / self.m

class ToyNeuralNetFn(): # Benchmark B: Toy Neural Network Quadratic Loss
    def f(self, x):
        x1, x2 = x[0], x[1]
        return (x1 - 1)**2 + 5*(x2 - 2)**2 + np.sin(x1)

    def df(self, x):
        x1, x2 = x[0], x[1]
        d1 = 2*(x1 - 1) + np.cos(x1)
        d2 = 10*(x2 - 2)
        return np.array([d1, d2])
    
class RosenbrockFn(): # Benchmark C: Rosenbrock Function
    def f(self, x):
        x1, x2 = x[0], x[1]
        return (1 - x1)**2 + 100*(x2 - x1**2)**2

    def df(self, x):
        x1, x2 = x[0], x[1]
        d1 = -2*(1 - x1) - 400*x1*(x2 - x1**2)
        d2 = 200*(x2 - x1**2)
        return np.array([d1, d2])

def polyakGradDescent(fn, x0, f_star=0.0, eps=1.0e-4, num_iters=120): # Polyak Step Size Gradient Descent
    x = np.array(x0, dtype=float)
    X = np.array([x.copy()])
    F = np.array([fn.f(x)])
    A = np.array([])
    for _ in range(num_iters):
        grad = fn.df(x)
        alpha = (fn.f(x) - f_star) / (np.sum(grad**2) + eps) # alpha_k = (f(x_k) - f_star) / (||grad f(x_k)||^2 + eps)
        x = x - alpha * grad
        X = np.append(X, [x.copy()], axis=0)
        F = np.append(F, fn.f(x))
        A = np.append(A, alpha)
    return (X, F, A)

fnA = LinearRegressionFn(m=1000, seed=0, noise_std=0.1) # Run Polyak on Benchmark A
(XA, FA, AA) = polyakGradDescent(fnA, x0=np.array([1.0, 1.0]), f_star=0.0, eps=1.0e-4, num_iters=120)

fnB = ToyNeuralNetFn() # Run Polyak on Benchmark B
(XB, FB, AB) = polyakGradDescent(fnB, x0=np.array([1.0, 1.0]), f_star=0.0, eps=1.0e-4, num_iters=120)

fnC = RosenbrockFn() # Run Polyak on Benchmark C
(XC, FC, AC) = polyakGradDescent(fnC, x0=np.array([1.0, 1.0]), f_star=0.0, eps=1.0e-3, num_iters=120)

# Q1 (a) (II)
def adagrad(fn, x0, alpha0=1.0, eps=1.0e-5, num_iters=120): # Adagrad with per-coordinate adaptive step sizes
    x = np.array(x0, dtype=float)
    X = np.array([x.copy()])
    F = np.array([fn.f(x)])
    G = np.zeros_like(x) # running sum of squared gradients
    A = np.empty((0, len(x))) # store per-coordinate step sizes
    for _ in range(num_iters):
        grad = fn.df(x)
        G = G + grad**2
        alpha = alpha0 / np.sqrt(G + eps) # alpha_k,i = alpha0 / sqrt(sum_{t<=k} grad_i^2 + eps)
        x = x - alpha * grad
        X = np.append(X, [x.copy()], axis=0)
        F = np.append(F, fn.f(x))
        A = np.append(A, [alpha.copy()], axis=0)
    return (X, F, A)

(XA_ada, FA_ada, AA_ada) = adagrad(fnA, x0=np.array([1.0, 1.0]), alpha0=1.8, eps=1.0e-5, num_iters=120) # Run Adagrad on Benchmark A
(XB_ada, FB_ada, AB_ada) = adagrad(fnB, x0=np.array([1.0, 1.0]), alpha0=1.2, eps=1.0e-5, num_iters=120) # Run Adagrad on Benchmark B
(XC_ada, FC_ada, AC_ada) = adagrad(fnC, x0=np.array([1.0, 1.0]), alpha0=0.45, eps=1.0e-5, num_iters=120) # Run Adagrad on Benchmark C

# Q1 (a) (III)
def rmsprop(fn, x0, alpha0=1.0, beta=0.9, eps=1.0e-5, num_iters=120): # RMSprop using exponentially weighted squared gradients
    x = np.array(x0, dtype=float)
    X = np.array([x.copy()])
    F = np.array([fn.f(x)])
    S = np.zeros_like(x) # running weighted sum of squared gradients
    A = np.empty((0, len(x))) # store per-coordinate step sizes
    for _ in range(num_iters):
        grad = fn.df(x)
        S = beta*S + (1 - beta)*(grad**2)
        alpha = alpha0 / (np.sqrt(S) + eps) # alpha_k,i = alpha0 / (sqrt(S_k,i) + eps)
        x = x - alpha * grad
        X = np.append(X, [x.copy()], axis=0)
        F = np.append(F, fn.f(x))
        A = np.append(A, [alpha.copy()], axis=0)
    return (X, F, A)

(XA_rms, FA_rms, AA_rms) = rmsprop(fnA, x0=np.array([1.0, 1.0]), alpha0=0.22, beta=0.9, eps=1.0e-5, num_iters=120)
(XB_rms, FB_rms, AB_rms) = rmsprop(fnB, x0=np.array([1.0, 1.0]), alpha0=0.14, beta=0.9, eps=1.0e-5, num_iters=120)
(XC_rms, FC_rms, AC_rms) = rmsprop(fnC, x0=np.array([1.0, 1.0]), alpha0=0.0035, beta=0.9, eps=1.0e-5, num_iters=120)

# Q1 (a) (IV)
def heavyBall(fn, x0, alpha=0.01, beta=0.9, num_iters=120): # Polyak Momentum / Heavy Ball
    x = np.array(x0, dtype=float)
    z = np.zeros_like(x)
    X = np.array([x.copy()])
    F = np.array([fn.f(x)])
    Z = np.empty((0, len(x))) # store momentum steps
    for _ in range(num_iters):
        grad = fn.df(x)
        z = beta*z + alpha*grad
        x = x - z
        X = np.append(X, [x.copy()], axis=0)
        F = np.append(F, fn.f(x))
        Z = np.append(Z, [z.copy()], axis=0)
    return (X, F, Z)

(XA_hb, FA_hb, ZA_hb) = heavyBall(fnA, x0=np.array([1.0, 1.0]), alpha=0.045, beta=0.88, num_iters=120)
(XB_hb, FB_hb, ZB_hb) = heavyBall(fnB, x0=np.array([1.0, 1.0]), alpha=0.035, beta=0.90, num_iters=120)
(XC_hb, FC_hb, ZC_hb) = heavyBall(fnC, x0=np.array([1.0, 1.0]), alpha=0.0008, beta=0.86, num_iters=120)

# Q1 (a) Constant step-size Gradient Descent baseline
def gradDescent(fn, x0, alpha=0.1, num_iters=120): 
    x = np.array(x0, dtype=float)
    X = np.array([x.copy()])
    F = np.array([fn.f(x)])
    for _ in range(num_iters):
        grad = fn.df(x)
        x = x - alpha*grad
        X = np.append(X, [x.copy()], axis=0)
        F = np.append(F, fn.f(x))
    return (X, F)

(XA_gd, FA_gd) = gradDescent(fnA, x0=np.array([1.0, 1.0]), alpha=0.08, num_iters=120)   # Benchmark A
(XB_gd, FB_gd) = gradDescent(fnB, x0=np.array([1.0, 1.0]), alpha=0.06, num_iters=120)   # Benchmark B
(XC_gd, FC_gd) = gradDescent(fnC, x0=np.array([1.0, 1.0]), alpha=0.0012, num_iters=120) # Benchmark C
