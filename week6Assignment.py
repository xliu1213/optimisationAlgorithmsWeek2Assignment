import numpy as np
import matplotlib.pyplot as plt

# Q1 (a)
class LinearRegressionFn:  # Linear regression loss L(theta) = (1/2m)||Xθ - y||^2
    def f(self, theta):  # loss function
        r = self.X @ theta - self.y
        return (1/(2*self.m)) * np.dot(r, r)
    def df(self, theta):  # gradient
        r = self.X @ theta - self.y
        return (1/self.m) * (self.X.T @ r)

def gradDescent(fn, x0, alpha=0.5, num_iters=80):  # gradient descent
    x = x0
    X = np.array([x])
    F = np.array([fn.f(x)])
    for _ in range(num_iters):
        step = alpha * fn.df(x)
        x = x - step
        X = np.append(X, [x], axis=0)
        F = np.append(F, fn.f(x))
    return (X, F)

m = 1000 # Generate synthetic data
X = np.random.normal(0, 1, (m, 2))
theta_star = np.array([3, 4])
sigma = 1
eps = np.random.normal(0, sigma, m)
y = X @ theta_star + eps

fn = LinearRegressionFn() # create function object
fn.X = X
fn.y = y
fn.m = m
theta0 = np.array([1.0, 1.0])
Theta, Loss = gradDescent(fn, theta0, alpha=0.5, num_iters=80) # run GD

plt.semilogy(Loss) # Plot 1: Loss vs iteration (log scale)
plt.xlabel('iteration')
plt.ylabel('L(theta)')
plt.title('Full Batch GD Loss')
plt.show()

t1 = np.linspace(0,5,100) # Plot 2: Contour + trajectory
t2 = np.linspace(0,6,100)
T1, T2 = np.meshgrid(t1, t2)
Z = np.zeros_like(T1)
for i in range(T1.shape[0]):
    for j in range(T1.shape[1]):
        theta = np.array([T1[i,j], T2[i,j]])
        Z[i,j] = fn.f(theta)
plt.contour(T1, T2, Z, levels=30)
plt.plot(Theta[:,0], Theta[:,1], 'ro-')
plt.xlabel('theta1')
plt.ylabel('theta2')
plt.title('GD trajectory')
plt.show()

# Q1 (b)
def miniBatchSGD(fn, theta0, alpha=0.5, batch_size=5, num_updates=400): # Mini-batch SGD with different batch sizes
    theta = theta0.copy()
    Theta = np.array([theta])
    Loss = np.array([fn.f(theta)])
    m = fn.m
    X = fn.X
    y = fn.y
    updates = 0
    while updates < num_updates:
        perm = np.random.permutation(m) # shuffle once per epoch
        Xs = X[perm]
        ys = y[perm]
        for i in range(0, m, batch_size): # go through mini-batches
            if updates >= num_updates:
                break
            Xb = Xs[i:i+batch_size]
            yb = ys[i:i+batch_size]
            r = Xb @ theta - yb # compute mini-batch gradient
            grad = (1/len(Xb)) * (Xb.T @ r)
            step = alpha * grad
            theta = theta - step
            Theta = np.append(Theta, [theta], axis=0)
            Loss = np.append(Loss, fn.f(theta))
            updates += 1
    return Theta, Loss
Theta_sgd5, Loss_sgd5 = miniBatchSGD(fn, theta0, alpha=0.5, batch_size=5, num_updates=400) # Run SGD with batch size 5
Theta_sgd20, Loss_sgd20 = miniBatchSGD(fn, theta0, alpha=0.5, batch_size=20, num_updates=400) # Run SGD with batch size 20
plt.semilogy(Loss, label='GD') # Loss Plot 
plt.semilogy(Loss_sgd5, label='SGD (b=5)')
plt.semilogy(Loss_sgd20, label='SGD (b=20)')
plt.xlabel('iteration')
plt.ylabel('L(theta)')
plt.title('Loss comparison: GD vs SGD')
plt.legend()
plt.show()
plt.contour(T1, T2, Z, levels=30) # Contour Plot 
plt.plot(Theta[:,0], Theta[:,1], 'r-', label='GD')
plt.plot(Theta_sgd5[:,0], Theta_sgd5[:,1], 'g-', label='SGD (b=5)')
plt.plot(Theta_sgd20[:,0], Theta_sgd20[:,1], 'b-', label='SGD (b=20)')
plt.xlabel('theta1')
plt.ylabel('theta2')
plt.title('Trajectory comparison')
plt.legend()
plt.show()

# Q1 (c)
Theta_sgd10, Loss_sgd10 = miniBatchSGD(fn, theta0, alpha=0.5, batch_size=10, num_updates=400) # Reuse miniBatchSGD for constant SGD with batch size 10
def NAG(fn, theta0, alpha=0.5, beta=0.9, batch_size=10, num_updates=400):
    theta = theta0.copy()
    v = np.zeros_like(theta)
    Theta = np.array([theta])
    Loss = np.array([fn.f(theta)])
    m = fn.m
    X = fn.X
    y = fn.y
    updates = 0
    while updates < num_updates:
        perm = np.random.permutation(m)
        Xs = X[perm]
        ys = y[perm]
        for i in range(0, m, batch_size):
            if updates >= num_updates:
                break
            Xb = Xs[i:i+batch_size]
            yb = ys[i:i+batch_size]
            theta_look = theta - beta * v # lookahead point
            r = Xb @ theta_look - yb
            grad = (1/len(Xb)) * (Xb.T @ r)
            v = beta * v + alpha * grad
            theta = theta - v
            Theta = np.append(Theta, [theta], axis=0)
            Loss = np.append(Loss, fn.f(theta))
            updates += 1
    return Theta, Loss

def Adagrad(fn, theta0, alpha=0.5, batch_size=10, num_updates=400, eps=1e-8):
    theta = theta0.copy()
    G = np.zeros_like(theta)
    Theta = np.array([theta])
    Loss = np.array([fn.f(theta)])
    m = fn.m
    X = fn.X
    y = fn.y
    updates = 0
    while updates < num_updates:
        perm = np.random.permutation(m)
        Xs = X[perm]
        ys = y[perm]
        for i in range(0, m, batch_size):
            if updates >= num_updates:
                break
            Xb = Xs[i:i+batch_size]
            yb = ys[i:i+batch_size]
            r = Xb @ theta - yb
            grad = (1/len(Xb)) * (Xb.T @ r)
            G = G + grad**2
            step = (alpha / (np.sqrt(G) + eps)) * grad
            theta = theta - step
            Theta = np.append(Theta, [theta], axis=0)
            Loss = np.append(Loss, fn.f(theta))
            updates += 1
    return Theta, Loss

Theta_nag, Loss_nag = NAG(fn, theta0, alpha=0.5, beta=0.9, batch_size=10, num_updates=400) # Run NAG and Adagrad
Theta_ada, Loss_ada = Adagrad(fn, theta0, alpha=0.5, batch_size=10, num_updates=400)

plt.semilogy(Loss_sgd10, label='SGD (b=10)') # Loss comparison plot
plt.semilogy(Loss_nag, label='NAG')
plt.semilogy(Loss_ada, label='Adagrad')
plt.xlabel('iteration')
plt.ylabel('L(theta)')
plt.title('Loss comparison: SGD vs NAG vs Adagrad')
plt.legend()
plt.show()
t1 = np.linspace(0,5,100) # Contour + trajectory plot
t2 = np.linspace(0,6,100)
T1, T2 = np.meshgrid(t1, t2)
Z = np.zeros_like(T1)
for i in range(T1.shape[0]):
    for j in range(T1.shape[1]):
        theta = np.array([T1[i,j], T2[i,j]])
        Z[i,j] = fn.f(theta)
plt.contour(T1, T2, Z, levels=30)
plt.plot(Theta_sgd10[:,0], Theta_sgd10[:,1], 'r-', label='SGD')
plt.plot(Theta_nag[:,0], Theta_nag[:,1], 'b-', label='NAG')
plt.plot(Theta_ada[:,0], Theta_ada[:,1], 'g-', label='Adagrad')
plt.xlabel('theta1')
plt.ylabel('theta2')
plt.title('Optimisation trajectories')
plt.legend()
plt.show()

# Q2 (a)
class ToyNNFn:  # Toy neural net loss
    def f(self, x):  # loss function J(x)
        x1, x2 = x
        yhat = x2 * np.tanh(x1 * self.u)
        r = yhat - self.y
        return (1/self.m) * 0.5 * np.dot(r, r)
    def df(self, x):  # gradient of J(x)
        x1, x2 = x
        tanh_term = np.tanh(x1 * self.u)
        yhat = x2 * tanh_term
        r = yhat - self.y
        dJ_dx1 = (1/self.m) * np.sum(r * (x2 * (1 - tanh_term**2) * self.u)) # derivatives
        dJ_dx2 = (1/self.m) * np.sum(r * tanh_term)
        return np.array([dJ_dx1, dJ_dx2])
m = 1000 # Generate synthetic data
u = np.random.uniform(-2, 2, m)
x_star = np.array([1, 3])
sigma = 0.05
eps = np.random.normal(0, sigma, m)
y = x_star[1] * np.tanh(x_star[0] * u) + eps
fn = ToyNNFn()
fn.u = u
fn.y = y
fn.m = m
x0 = np.array([1.0, 1.0])
Theta2, Loss2 = gradDescent(fn, x0, alpha=0.75, num_iters=500)
plt.semilogy(Loss2) # Plot 1: Loss vs iteration
plt.xlabel('iteration')
plt.ylabel('J(x)')
plt.title('Toy NN GD Loss')
plt.show()
t1 = np.linspace(0,2,100) # Plot 2: Contour + trajectory
t2 = np.linspace(0,6,100)
T1, T2 = np.meshgrid(t1, t2)
Z = np.zeros_like(T1)
for i in range(T1.shape[0]):
    for j in range(T1.shape[1]):
        x = np.array([T1[i,j], T2[i,j]])
        Z[i,j] = fn.f(x)
plt.contour(T1, T2, Z, levels=30)
plt.plot(Theta2[:,0], Theta2[:,1], 'ro-')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Toy NN GD trajectory')
plt.show()

# Q2 (b)
def miniBatchSGD_ToyNN(fn, x0, alpha=0.75, batch_size=5, num_updates=500):
    x = x0.copy()
    Theta = np.array([x])
    Loss = np.array([fn.f(x)])
    m = fn.m
    u = fn.u
    y = fn.y
    updates = 0
    while updates < num_updates:
        perm = np.random.permutation(m)  # shuffle once per epoch
        us = u[perm]
        ys = y[perm]
        for i in range(0, m, batch_size):  # go through mini-batches
            if updates >= num_updates:
                break
            ub = us[i:i+batch_size]
            yb = ys[i:i+batch_size]
            x1, x2 = x
            tanh_term = np.tanh(x1 * ub)
            yhat = x2 * tanh_term
            r = yhat - yb
            dJ_dx1 = (1/len(ub)) * np.sum(r * (x2 * (1 - tanh_term**2) * ub))
            dJ_dx2 = (1/len(ub)) * np.sum(r * tanh_term)
            grad = np.array([dJ_dx1, dJ_dx2])
            step = alpha * grad
            x = x - step
            Theta = np.append(Theta, [x], axis=0)
            Loss = np.append(Loss, fn.f(x))
            updates += 1
    return Theta, Loss
Theta_sgd5_q2, Loss_sgd5_q2 = miniBatchSGD_ToyNN(fn, x0, alpha=0.75, batch_size=5, num_updates=500)
Theta_sgd20_q2, Loss_sgd20_q2 = miniBatchSGD_ToyNN(fn, x0, alpha=0.75, batch_size=20, num_updates=500)
plt.semilogy(Loss2, label='GD') # Loss comparison plot
plt.semilogy(Loss_sgd5_q2, label='SGD (b=5)')
plt.semilogy(Loss_sgd20_q2, label='SGD (b=20)')
plt.xlabel('iteration')
plt.ylabel('J(x)')
plt.title('Toy NN Loss comparison')
plt.legend()
plt.show()
plt.contour(T1, T2, Z, levels=30) # Contour trajectory comparison
plt.plot(Theta2[:,0], Theta2[:,1], 'r-', label='GD')
plt.plot(Theta_sgd5_q2[:,0], Theta_sgd5_q2[:,1], 'g-', label='SGD (b=5)')
plt.plot(Theta_sgd20_q2[:,0], Theta_sgd20_q2[:,1], 'b-', label='SGD (b=20)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Toy NN trajectory comparison')
plt.legend()
plt.show()

# Q2 (c)
Theta_sgd10_q2, Loss_sgd10_q2 = miniBatchSGD_ToyNN(fn, x0, alpha=0.75, batch_size=10, num_updates=500) # Reuse miniBatchSGD_ToyNN for constant SGD baseline
def NAG_ToyNN(fn, x0, alpha=0.75, beta=0.9, batch_size=10, num_updates=500):
    x = x0.copy()
    v = np.zeros_like(x)
    Theta = np.array([x])
    Loss = np.array([fn.f(x)])
    m = fn.m
    u = fn.u
    y = fn.y
    updates = 0
    while updates < num_updates:
        perm = np.random.permutation(m)
        us = u[perm]
        ys = y[perm]
        for i in range(0, m, batch_size):
            if updates >= num_updates:
                break
            ub = us[i:i+batch_size]
            yb = ys[i:i+batch_size]
            x_look = x - beta * v # lookahead point
            x1, x2 = x_look
            tanh_term = np.tanh(x1 * ub)
            yhat = x2 * tanh_term
            r = yhat - yb
            dJ_dx1 = (1/len(ub)) * np.sum(r * (x2 * (1 - tanh_term**2) * ub))
            dJ_dx2 = (1/len(ub)) * np.sum(r * tanh_term)
            grad = np.array([dJ_dx1, dJ_dx2])
            v = beta * v + alpha * grad
            x = x - v
            Theta = np.append(Theta, [x], axis=0)
            Loss = np.append(Loss, fn.f(x))
            updates += 1
    return Theta, Loss

def Adagrad_ToyNN(fn, x0, alpha=0.75, batch_size=10, num_updates=500, eps=1e-8):
    x = x0.copy()
    G = np.zeros_like(x)
    Theta = np.array([x])
    Loss = np.array([fn.f(x)])
    m = fn.m
    u = fn.u
    y = fn.y
    updates = 0
    while updates < num_updates:
        perm = np.random.permutation(m)
        us = u[perm]
        ys = y[perm]
        for i in range(0, m, batch_size):
            if updates >= num_updates:
                break
            ub = us[i:i+batch_size]
            yb = ys[i:i+batch_size]
            x1, x2 = x
            tanh_term = np.tanh(x1 * ub)
            yhat = x2 * tanh_term
            r = yhat - yb
            dJ_dx1 = (1/len(ub)) * np.sum(r * (x2 * (1 - tanh_term**2) * ub))
            dJ_dx2 = (1/len(ub)) * np.sum(r * tanh_term)
            grad = np.array([dJ_dx1, dJ_dx2])
            G = G + grad**2
            step = (alpha / (np.sqrt(G) + eps)) * grad
            x = x - step
            Theta = np.append(Theta, [x], axis=0)
            Loss = np.append(Loss, fn.f(x))
            updates += 1
    return Theta, Loss

Theta_nag_q2, Loss_nag_q2 = NAG_ToyNN(fn, x0, alpha=0.75, beta=0.9, batch_size=10, num_updates=500) # Run optimisers
Theta_ada_q2, Loss_ada_q2 = Adagrad_ToyNN(fn, x0, alpha=0.75, batch_size=10, num_updates=500)

plt.semilogy(Loss_sgd10_q2, label='SGD (b=10)') # Loss comparison plot
plt.semilogy(Loss_nag_q2, label='NAG')
plt.semilogy(Loss_ada_q2, label='Adagrad')
plt.xlabel('iteration')
plt.ylabel('J(x)')
plt.title('Toy NN Loss comparison: SGD vs NAG vs Adagrad')
plt.legend()
plt.show()
plt.contour(T1, T2, Z, levels=30) # Contour trajectory plot
plt.plot(Theta_sgd10_q2[:,0], Theta_sgd10_q2[:,1], 'r-', label='SGD')
plt.plot(Theta_nag_q2[:,0], Theta_nag_q2[:,1], 'b-', label='NAG')
plt.plot(Theta_ada_q2[:,0], Theta_ada_q2[:,1], 'g-', label='Adagrad')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Toy NN optimisation trajectories')
plt.legend()
plt.show()

# Q3 (a)
class RosenbrockFn:  # Rosenbrock function
    def f(self, x):
        x1, x2 = x
        return (1 - x1)**2 + 100 * (x2 - x1**2)**2
    def df(self, x):  # gradient of Rosenbrock function
        x1, x2 = x
        df_dx1 = -2*(1 - x1) - 400*x1*(x2 - x1**2)
        df_dx2 = 200*(x2 - x1**2)
        return np.array([df_dx1, df_dx2])
fn = RosenbrockFn()
x0 = np.array([-1.0, 1.0])  # initial point given in assignment
Theta3, Loss3 = gradDescent(fn, x0, alpha=1e-3, num_iters=200)
plt.semilogy(Loss3) # Plot: f(x_t) vs iteration (log-scale)
plt.xlabel('iteration')
plt.ylabel('f(x)')
plt.title('Rosenbrock GD Loss')
plt.show()

# Q3 (b)
def finite_grad(fn, x, h=1e-5): # Finite difference gradient
    n = len(x)
    g = np.zeros(n)
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1
        g[i] = (fn.f(x + h*e) - fn.f(x - h*e)) / (2*h)
    return g

def finite_hessian(fn, x, h=1e-5): # Finite difference Hessian
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = 1
            ej[j] = 1
            H[i, j] = (
                fn.f(x + h*ei + h*ej)
                - fn.f(x + h*ei - h*ej)
                - fn.f(x - h*ei + h*ej)
                + fn.f(x - h*ei - h*ej)
            ) / (4 * h**2)
    return H

def newtonMethod(fn, x0, alpha=0.7, num_iters=200): # Newton method using finite differences
    x = x0.copy()
    Theta = np.array([x])
    Loss = np.array([fn.f(x)])
    for _ in range(num_iters):
        g = finite_grad(fn, x)
        H = finite_hessian(fn, x)
        p = np.linalg.solve(H, g)
        x = x - alpha * p
        Theta = np.append(Theta, [x], axis=0)
        Loss = np.append(Loss, fn.f(x))
    return Theta, Loss

Theta_newton, Loss_newton = newtonMethod(fn, x0, alpha=0.7, num_iters=200) # Run Newton method

plt.semilogy(Loss_newton)
plt.xlabel('iteration')
plt.ylabel('f(x)')
plt.title('Rosenbrock Newton Loss')
plt.show()

# Q3 (c)
def dampedNewtonMethod(fn, x0, alpha0=1.0, rho=0.5, K=20, num_iters=200, lam=1e-6):
    x = x0.copy()
    Theta = np.array([x])
    Loss = np.array([fn.f(x)])
    for _ in range(num_iters):
        ft = fn.f(x)
        g = finite_grad(fn, x)
        H = finite_hessian(fn, x)
        p = np.linalg.solve(H + lam * np.eye(len(x)), g) # Solve (H + lambda I) p = g
        alpha = alpha0
        accepted = False
        for _ in range(K):
            xnew = x - alpha * p
            if fn.f(xnew) < ft:
                accepted = True
                break
            alpha = rho * alpha
        if not accepted:
            xnew = x - 1e-4 * g  # fallback step from hint
        x = xnew
        Theta = np.append(Theta, [x], axis=0)
        Loss = np.append(Loss, fn.f(x))
    return Theta, Loss

Theta_damped, Loss_damped = dampedNewtonMethod(fn, x0, alpha0=1.0, rho=0.5, K=20, num_iters=200)

plt.semilogy(Loss3, label='GD') # Plot (I): loss comparison
plt.semilogy(Loss_newton, label='Newton')
plt.semilogy(Loss_damped, label='Damped Newton')
plt.xlabel('iteration')
plt.ylabel('f(x)')
plt.title('Rosenbrock: GD vs Newton vs Damped Newton')
plt.legend()
plt.show()
t1 = np.linspace(-1.5, 1.5, 400) # Plot (II): contour plot with trajectories
t2 = np.linspace(-0.5, 1.5, 400)
T1, T2 = np.meshgrid(t1, t2)
Z = np.zeros_like(T1)
for i in range(T1.shape[0]):
    for j in range(T1.shape[1]):
        x = np.array([T1[i, j], T2[i, j]])
        Z[i, j] = fn.f(x)
plt.contour(T1, T2, Z, levels=np.logspace(-1, 3, 20))
plt.plot(Theta3[::5, 0], Theta3[::5, 1], 'r.-', label='GD (subsampled)')
plt.plot(Theta_newton[:, 0], Theta_newton[:, 1], 'b.-', label='Newton')
plt.plot(Theta_damped[:, 0], Theta_damped[:, 1], 'g.-', label='Damped Newton')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Rosenbrock trajectories')
plt.legend()
plt.show()