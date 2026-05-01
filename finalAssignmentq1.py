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

# # Q1 (c)
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

# Q2 (I): Nesterov Momentum / Acceleration
def nesterov(fn, x0, alpha=0.01, beta_max=0.9, num_iters=150):
    x = np.array(x0, dtype=float)
    z = np.zeros_like(x)
    X = np.array([x.copy()])
    F = np.array([fn.f(x)])
    B = np.array([]) # store beta values
    for k in range(num_iters):
        t = k + 1
        beta = min((t - 1) / (t + 2), beta_max)
        grad = fn.df(x + beta*z)
        z = beta*z - alpha*grad
        x = x + z
        X = np.append(X, [x.copy()], axis=0)
        F = np.append(F, fn.f(x))
        B = np.append(B, beta)
    return (X, F, B)

(XA_nag, FA_nag, BA_nag) = nesterov(fnA, x0=np.array([1.0, 1.0]), alpha=0.06, beta_max=0.90, num_iters=150) # Run Nesterov on Benchmark A, B, and C
(XB_nag, FB_nag, BB_nag) = nesterov(fnB, np.array([1.0, 1.0]), 0.035, 0.92, 150)
(XC_nag, FC_nag, BC_nag) = nesterov(fnC, np.array([1.0, 1.0]), 0.0007, 0.90, 150)

# Q2 (II): Adam
def adam(fn, x0, alpha=0.01, beta1=0.9, beta2=0.999, eps=1.0e-8, num_iters=150):
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x) # first moment estimate
    v = np.zeros_like(x) # second moment estimate
    X = np.array([x.copy()])
    F = np.array([fn.f(x)])
    for k in range(num_iters):
        t = k + 1
        grad = fn.df(x)
        m = beta1*m + (1 - beta1)*grad
        v = beta2*v + (1 - beta2)*(grad**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x = x - alpha * m_hat / (np.sqrt(v_hat) + eps)
        X = np.append(X, [x.copy()], axis=0)
        F = np.append(F, fn.f(x))
    return (X, F)

(XA_adam, FA_adam) = adam(fnA, np.array([1.0, 1.0]), 0.12, 0.82, 0.999, 1.0e-8, 150) # Run Adam on Benchmark A, B, and C
(XB_adam, FB_adam) = adam(fnB, np.array([1.0, 1.0]), 0.08, 0.80, 0.999, 1.0e-8, 150)
(XC_adam, FC_adam) = adam(fnC, np.array([1.0, 1.0]), 0.006, 0.80, 0.999, 1.0e-8, 150)

# Q2 (III): Mini-Batch Stochastic Gradient Descent
def miniBatchSGD(fn, theta0, alpha=0.06, batch_size=5, epochs=50):
    theta = theta0.copy()
    Theta = np.array([theta])
    Loss = np.array([fn.f(theta)])
    m = fn.m
    X = fn.X
    y = fn.y
    for _ in range(epochs):
        perm = np.random.permutation(m) # shuffle once per epoch
        Xs = X[perm]
        ys = y[perm]
        for i in range(0, m, batch_size):
            Xb = Xs[i:i+batch_size]
            yb = ys[i:i+batch_size]
            r = Xb @ theta - yb
            grad = (1/len(Xb)) * (Xb.T @ r)
            theta = theta - alpha*grad
            Theta = np.append(Theta, [theta.copy()], axis=0)
            Loss = np.append(Loss, fn.f(theta))
    return Theta, Loss

(XA_sgd_b5, FA_sgd_b5) = miniBatchSGD(fnA, np.array([1.0, 1.0]), 0.06, 5, 50) # Run Mini-Batch SGD on Benchmark A
(XA_sgd_b40, FA_sgd_b40) = miniBatchSGD(fnA, np.array([1.0, 1.0]), 0.06, 40, 50)

# Q2 (IV): SGD with Noise
fnA_noisy = LinearRegressionFn(m=1000, seed=0, noise_std=0.6) # original noise_std=0.1, increased by factor 6
(XA_noisy_sgd_b5, FA_noisy_sgd_b5) = miniBatchSGD(fnA_noisy, np.array([1.0, 1.0]), 0.06, 5, 50) # Run noisy Mini-Batch SGD on Benchmark A
(XA_noisy_sgd_b40, FA_noisy_sgd_b40) = miniBatchSGD(fnA_noisy, np.array([1.0, 1.0]), 0.06, 40, 50)

# Q2 constant step-size Gradient Descent baseline
(XA_gd_q2, FA_gd_q2) = gradDescent(fnA, np.array([1.0, 1.0]), 0.08, 150)
(XB_gd_q2, FB_gd_q2) = gradDescent(fnB, np.array([1.0, 1.0]), 0.06, 150)
(XC_gd_q2, FC_gd_q2) = gradDescent(fnC, np.array([1.0, 1.0]), 0.0012, 150)

# Q2 (b): Objective value versus iteration for Nesterov, Adam, and GD baseline
iters_150 = np.arange(151)
plt.figure() # Benchmark A: Linear Regression
plt.plot(iters_150, FA_nag, label='Nesterov')
plt.plot(iters_150, FA_adam, label='Adam')
plt.plot(iters_150, FA_gd_q2, label='GD baseline')
plt.xlabel('Iteration')
plt.ylabel('Objective value')
plt.title('Benchmark A: Objective value vs iteration')
plt.legend()
plt.grid(True)
plt.show()

plt.figure() # Benchmark B: Toy Neural Network
plt.plot(iters_150, FB_nag, label='Nesterov')
plt.plot(iters_150, FB_adam, label='Adam')
plt.plot(iters_150, FB_gd_q2, label='GD baseline')
plt.xlabel('Iteration')
plt.ylabel('Objective value')
plt.title('Benchmark B: Objective value vs iteration')
plt.legend()
plt.grid(True)
plt.show()

plt.figure() # Benchmark C: Rosenbrock
plt.plot(iters_150, FC_nag, label='Nesterov')
plt.plot(iters_150, FC_adam, label='Adam')
plt.plot(iters_150, FC_gd_q2, label='GD baseline')
plt.xlabel('Iteration')
plt.ylabel('Objective value')
plt.title('Benchmark C: Objective value vs iteration')
plt.legend()
plt.grid(True)
plt.show()

# Q2 (c): Contour plots with optimisation trajectories
plotContourWithTrajectories( # Benchmark B: Toy Neural Network
    fnB,
    x1_range=(-1.5, 2.5),
    x2_range=(0.5, 3.5),
    trajectories=[XB_nag, XB_adam, XB_gd_q2],
    labels=['Nesterov', 'Adam', 'GD baseline'],
    title='Benchmark B: Contour plot with optimisation trajectories'
)

plotContourWithTrajectories( # Benchmark C: Rosenbrock
    fnC,
    x1_range=(-1.5, 1.5),
    x2_range=(-0.5, 2.0),
    trajectories=[XC_nag, XC_adam, XC_gd_q2],
    labels=['Nesterov', 'Adam', 'GD baseline'],
    title='Benchmark C: Contour plot with optimisation trajectories'
)

# Q2 (d): Mini-Batch SGD loss comparison, batch size 5 versus batch size 40
plt.figure()
plt.semilogy(FA_sgd_b5, label='SGD, batch size 5')
plt.semilogy(FA_sgd_b40, label='SGD, batch size 40')
plt.xlabel('Update')
plt.ylabel('Loss')
plt.title('Benchmark A: Mini-Batch SGD loss comparison')
plt.legend()
plt.grid(True)
plt.show()

# Q2 (e): Effect of increased output noise on Mini-Batch SGD
plt.figure()
plt.semilogy(FA_sgd_b5, label='Original noise, batch size 5')
plt.semilogy(FA_sgd_b40, label='Original noise, batch size 40')
plt.semilogy(FA_noisy_sgd_b5, label='Higher noise, batch size 5')
plt.semilogy(FA_noisy_sgd_b40, label='Higher noise, batch size 40')
plt.xlabel('Update')
plt.ylabel('Loss')
plt.title('Benchmark A: Mini-Batch SGD under original and higher noise')
plt.legend()
plt.grid(True)
plt.show()

# Q3 (I) and Q3 (II): First-order and second-order local approximation
class QuarticFn(): # one-dimensional test function g(x) = x^4
    def f(self, x):
        return x**4

    def df(self, x):
        return 4*x**3

    def ddf(self, x):
        return 12*x**2

def firstOrderApprox(fn, x, x0): # g(x0) + g'(x0)(x - x0)
    return fn.f(x0) + fn.df(x0)*(x - x0)

def secondOrderApprox(fn, x, x0): # g(x0) + g'(x0)(x - x0) + 0.5*g''(x0)*(x - x0)^2
    return fn.f(x0) + fn.df(x0)*(x - x0) + 0.5*fn.ddf(x0)*(x - x0)**2

fnQ3 = QuarticFn()
x0_q3 = 0.25
xx = np.arange(-1, 1.01, 0.01)
g = fnQ3.f(xx)
g_linear = firstOrderApprox(fnQ3, xx, x0_q3)
g_quadratic = secondOrderApprox(fnQ3, xx, x0_q3)

plt.figure()
plt.plot(xx, g, label='Original function g(x) = x^4')
plt.plot(xx, g_linear, label='First-order approximation at x0 = 0.25')
plt.scatter([x0_q3], [fnQ3.f(x0_q3)], marker='o', label='Expansion point x0')
plt.xlabel('x')
plt.ylabel('g(x)')
plt.title('Q3 (I): First-order local approximation of g(x) = x^4')
plt.legend()
plt.grid(True)
plt.show()

# Q3 (II): Newton's Method with Hessian damping
def hessianLinearRegression(fn, x): # x unused, but keeps interface consistent
    return (fn.X.T @ fn.X) / fn.m

def hessianToyNeuralNet(fn, x):
    x1 = x[0]
    return np.array([
        [2 - np.sin(x1), 0],
        [0, 10]
    ])

def hessianRosenbrock(fn, x):
    x1, x2 = x[0], x[1]
    h11 = 2 - 400*x2 + 1200*x1**2
    h12 = -400*x1
    h22 = 200
    return np.array([
        [h11, h12],
        [h12, h22]
    ])

def newtonMethod(fn, hessian_fn, x0, alpha=1.0, damping=1.0e-8, num_iters=20):
    x = np.array(x0, dtype=float)
    X = np.array([x.copy()])
    F = np.array([fn.f(x)])
    U = np.array([])
    for _ in range(num_iters):
        grad = fn.df(x)
        H = hessian_fn(fn, x)
        H_damped = H + damping*np.eye(len(x))
        step = np.linalg.solve(H_damped, grad)
        x = x - alpha*step
        X = np.append(X, [x.copy()], axis=0)
        F = np.append(F, fn.f(x))
        U = np.append(U, np.linalg.norm(alpha*step))
    return (X, F, U)

(XA_newton, FA_newton, UA_newton) = newtonMethod(fnA, hessianLinearRegression, np.array([1.0, 1.0]), alpha=1.0, damping=1.0e-8, num_iters=20)
(XB_newton, FB_newton, UB_newton) = newtonMethod(fnB, hessianToyNeuralNet, np.array([1.0, 1.0]), alpha=0.85, damping=1.0e-8, num_iters=20)
(XC_newton, FC_newton, UC_newton) = newtonMethod(fnC, hessianRosenbrock, np.array([1.0, 1.0]), alpha=0.22, damping=1.0e-8, num_iters=20)
(XA_gd_q3, FA_gd_q3) = gradDescent(fnA, np.array([1.0, 1.0]), alpha=0.08, num_iters=80)
(XB_gd_q3, FB_gd_q3) = gradDescent(fnB, np.array([1.0, 1.0]), alpha=0.06, num_iters=80)
(XC_gd_q3, FC_gd_q3) = gradDescent(fnC, np.array([1.0, 1.0]), alpha=0.001, num_iters=80)

# Q3 (a): First-order and second-order approximations
plt.figure()
plt.plot(xx, g, label='Original function g(x) = x^4')
plt.plot(xx, g_linear, label='First-order approximation at x0 = 0.25')
plt.plot(xx, g_quadratic, label='Second-order approximation at x0 = 0.25')
plt.scatter([x0_q3], [fnQ3.f(x0_q3)], marker='o', label='Expansion point x0')
plt.xlabel('x')
plt.ylabel('g(x)')
plt.title('Q3 (a): Original function with first-order and second-order approximations')
plt.legend()
plt.grid(True)
plt.show()

# Q3 (c): Compare Newton's Method with Gradient Descent
plt.figure()
plt.plot(np.arange(81), FA_gd_q3, label='GD baseline')
plt.plot(np.arange(21), FA_newton, label='Newton')
plt.xlabel('Iteration')
plt.ylabel('Objective value')
plt.title('Q3 (c): Benchmark A Newton vs GD')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(np.arange(81), FB_gd_q3, label='GD baseline')
plt.plot(np.arange(21), FB_newton, label='Newton')
plt.xlabel('Iteration')
plt.ylabel('Objective value')
plt.title('Q3 (c): Benchmark B Newton vs GD')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(np.arange(81), FC_gd_q3, label='GD baseline')
plt.plot(np.arange(21), FC_newton, label='Newton')
plt.xlabel('Iteration')
plt.ylabel('Objective value')
plt.title('Q3 (c): Benchmark C Newton vs GD')
plt.legend()
plt.grid(True)
plt.show()

# Q3 (d): Contour trajectories for Benchmark B and Benchmark C
plotContourWithTrajectories(
    fnB,
    x1_range=(-1.5, 2.5),
    x2_range=(0.5, 3.5),
    trajectories=[XB_gd_q3, XB_newton],
    labels=['GD baseline', 'Newton'],
    title='Q3 (d): Benchmark B Newton vs GD contour trajectories'
)

plotContourWithTrajectories(
    fnC,
    x1_range=(-1.5, 1.5),
    x2_range=(-0.5, 2.0),
    trajectories=[XC_gd_q3, XC_newton],
    labels=['GD baseline', 'Newton'],
    title='Q3 (d): Benchmark C Newton vs GD contour trajectories'
)

# Q3 (e): Update magnitude versus iteration
def updateMagnitude(X):
    U = np.array([])
    for k in range(1, len(X)):
        U = np.append(U, np.linalg.norm(X[k] - X[k-1]))
    return U

UA_gd_q3 = updateMagnitude(XA_gd_q3)
UB_gd_q3 = updateMagnitude(XB_gd_q3)
UC_gd_q3 = updateMagnitude(XC_gd_q3)

plt.figure()
plt.plot(np.arange(1, 81), UA_gd_q3, label='GD baseline')
plt.plot(np.arange(1, 21), UA_newton, label='Newton')
plt.xlabel('Iteration')
plt.ylabel('Update magnitude')
plt.title('Q3 (e): Benchmark A update magnitude')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(np.arange(1, 81), UB_gd_q3, label='GD baseline')
plt.plot(np.arange(1, 21), UB_newton, label='Newton')
plt.xlabel('Iteration')
plt.ylabel('Update magnitude')
plt.title('Q3 (e): Benchmark B update magnitude')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(np.arange(1, 81), UC_gd_q3, label='GD baseline')
plt.plot(np.arange(1, 21), UC_newton, label='Newton')
plt.xlabel('Iteration')
plt.ylabel('Update magnitude')
plt.title('Q3 (e): Benchmark C update magnitude')
plt.legend()
plt.grid(True)
plt.show()
