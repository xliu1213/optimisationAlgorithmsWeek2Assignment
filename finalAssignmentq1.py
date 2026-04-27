import numpy as np
import matplotlib.pyplot as plt

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

# Q1 (b) 
iters_120 = np.arange(121) # includes iteration 0 up to 120
plt.figure() # Benchmark A: Linear Regression
plt.plot(iters_120, FA, label='Polyak')
plt.plot(iters_120, FA_ada, label='Adagrad')
plt.plot(iters_120, FA_rms, label='RMSprop')
plt.plot(iters_120, FA_hb, label='Heavy Ball')
plt.plot(iters_120, FA_gd, label='GD baseline')
plt.xlabel('Iteration')
plt.ylabel('Objective value')
plt.title('Benchmark A: Objective value vs iteration')
plt.legend()
plt.grid(True)
plt.show()

plt.figure() # Benchmark B: Toy Neural Network
plt.plot(iters_120, FB, label='Polyak')
plt.plot(iters_120, FB_ada, label='Adagrad')
plt.plot(iters_120, FB_rms, label='RMSprop')
plt.plot(iters_120, FB_hb, label='Heavy Ball')
plt.plot(iters_120, FB_gd, label='GD baseline')
plt.xlabel('Iteration')
plt.ylabel('Objective value')
plt.title('Benchmark B: Objective value vs iteration')
plt.legend()
plt.grid(True)
plt.show()

plt.figure() # Benchmark C: Rosenbrock
plt.plot(iters_120, FC, label='Polyak')
plt.plot(iters_120, FC_ada, label='Adagrad')
plt.plot(iters_120, FC_rms, label='RMSprop')
plt.plot(iters_120, FC_hb, label='Heavy Ball')
plt.plot(iters_120, FC_gd, label='GD baseline')
plt.xlabel('Iteration')
plt.ylabel('Objective value')
plt.title('Benchmark C: Objective value vs iteration')
plt.legend()
plt.grid(True)
plt.show()

# Q1 (c)
def plotContourWithTrajectories(fn, x1_range, x2_range, trajectories, labels, title):
    xx1 = np.linspace(x1_range[0], x1_range[1], 200)
    xx2 = np.linspace(x2_range[0], x2_range[1], 200)
    X1, X2 = np.meshgrid(xx1, xx2)
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = fn.f(np.array([X1[i, j], X2[i, j]]))
    plt.figure()
    plt.contour(X1, X2, Z, levels=30)
    for X, label in zip(trajectories, labels):
        plt.plot(X[:, 0], X[:, 1], marker='o', markersize=2, linewidth=1.5, label=label)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

plotContourWithTrajectories( # Benchmark B: Toy Neural Network
    fnB,
    x1_range=(-1.5, 2.5),
    x2_range=(0.5, 3.5),
    trajectories=[XB, XB_ada, XB_rms, XB_hb, XB_gd],
    labels=['Polyak', 'Adagrad', 'RMSprop', 'Heavy Ball', 'GD baseline'],
    title='Benchmark B: Contour plot with optimisation trajectories'
)

plotContourWithTrajectories( # Benchmark C: Rosenbrock
    fnC,
    x1_range=(-1.5, 1.5),
    x2_range=(-0.5, 2.0),
    trajectories=[XC],
    labels=['Polyak'],
    title='Benchmark C: Contour plot with optimisation trajectories'
)

plotContourWithTrajectories( # Benchmark C: Rosenbrock
    fnC,
    x1_range=(-1.5, 1.5),
    x2_range=(-0.5, 2.0),
    trajectories=[XC, XC_ada, XC_rms, XC_hb, XC_gd],
    labels=['Polyak', 'Adagrad', 'RMSprop', 'Heavy Ball', 'GD baseline'],
    title='Benchmark C: Contour plot with optimisation trajectories'
)

# Q1 (d)
iters_adapt = np.arange(1, 121) # adaptive step sizes are stored for iterations 1,...,120
plt.figure() # Benchmark A: Linear Regression
plt.plot(iters_adapt, AA, label='Polyak')
plt.plot(iters_adapt, AA_ada[:, 0], label='Adagrad (coord 1)')
plt.plot(iters_adapt, AA_ada[:, 1], label='Adagrad (coord 2)')
plt.plot(iters_adapt, AA_rms[:, 0], label='RMSprop (coord 1)')
plt.plot(iters_adapt, AA_rms[:, 1], label='RMSprop (coord 2)')
plt.xlabel('Iteration')
plt.ylabel('Adaptive step size')
plt.title('Benchmark A: Adaptive step-size evolution')
plt.legend()
plt.grid(True)
plt.show()

plt.figure() # Benchmark B: Toy Neural Network
plt.plot(iters_adapt, AB, label='Polyak')
plt.plot(iters_adapt, AB_ada[:, 0], label='Adagrad (coord 1)')
plt.plot(iters_adapt, AB_ada[:, 1], label='Adagrad (coord 2)')
plt.plot(iters_adapt, AB_rms[:, 0], label='RMSprop (coord 1)')
plt.plot(iters_adapt, AB_rms[:, 1], label='RMSprop (coord 2)')
plt.xlabel('Iteration')
plt.ylabel('Adaptive step size')
plt.title('Benchmark B: Adaptive step-size evolution')
plt.legend()
plt.grid(True)
plt.show()

plt.figure() # Benchmark C: Rosenbrock
plt.plot(iters_adapt, AC, label='Polyak')
plt.plot(iters_adapt, AC_ada[:, 0], label='Adagrad (coord 1)')
plt.plot(iters_adapt, AC_ada[:, 1], label='Adagrad (coord 2)')
plt.plot(iters_adapt, AC_rms[:, 0], label='RMSprop (coord 1)')
plt.plot(iters_adapt, AC_rms[:, 1], label='RMSprop (coord 2)')
plt.xlabel('Iteration')
plt.ylabel('Adaptive step size')
plt.title('Benchmark C: Adaptive step-size evolution')
plt.legend()
plt.grid(True)
plt.show()
