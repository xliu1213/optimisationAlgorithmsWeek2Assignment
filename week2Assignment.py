import sympy
import numpy as np
import matplotlib.pyplot as plt

# ----- Q1(a): symbolic derivative -----
x = sympy.symbols('x', real=True)
f = x**4
dfdx = sympy.diff(f, x)
print(dfdx)

# ----- Q1(b): evaluate derivative + finite difference on [-2,2] -----
f_num = sympy.lambdify(x, f)
dfdx_num = sympy.lambdify(x, dfdx)
xx = np.linspace(-2, 2, 1000)
delta = 0.01
d_exact = dfdx_num(xx)
d_fd = (f_num(xx + delta) - f_num(xx)) / delta
print("First 5 exact derivative values:", d_exact[:5])
print("First 5 finite-diff values:", d_fd[:5])
plt.plot(xx, d_exact)
plt.xlabel('x'); plt.ylabel('exact derivative')
plt.show()
plt.plot(xx, d_fd)
plt.xlabel('x'); plt.ylabel('forward finite-difference approximation with delta = 0.01')
plt.show()

# ----- Q1(c): vary delta in [0.001, 1] and compute MAE -----
deltas = np.linspace(0.001, 1, 200)
mae = np.zeros(len(deltas))
for i in range(len(deltas)): # exact derivative values are fixed, so reuse d_exact
    delta = deltas[i]
    d_fd = (f_num(xx + delta) - f_num(xx)) / delta
    mae[i] = np.mean(np.abs(d_fd - d_exact))
print("First 5 delta values:", deltas[:5])
print("First 5 MAE values:", mae[:5])
plt.plot(deltas, mae) # Plot MAE vs delta
plt.xlabel('delta')
plt.ylabel('MAE')
plt.title('Mean Absolute Error vs delta (forward finite difference)')
plt.show()

# ----- Q1(d) + Q1(e): gradient descent for f(x)=x^4 with different alphas -----
class QuarticFn():
    def f(self, x):
        return x**4  # function value f(x)
    def df(self, x):
        return 4*x**3  # derivative of f(x)
def gradDescent(fn, x0, alpha=0.15, num_iters=50):
    x = x0
    X = np.array([x])
    F = np.array([fn.f(x)])
    for _ in range(num_iters):
        step = alpha * fn.df(x)
        x = x - step
        if abs(x) > 1e6:
            print("Diverged (x too large). Stopping early.")
            break
        X = np.append(X, [x], axis=0)
        F = np.append(F, fn.f(x))
    return (X, F)
fn_gd = QuarticFn()
(X1, F1) = gradDescent(fn_gd, x0=1, alpha=0.05)
(X2, F2) = gradDescent(fn_gd, x0=1, alpha=0.5)
(X3, F3) = gradDescent(fn_gd, x0=1, alpha=1.2)
print("Final x (alpha=0.05):", X1[-1], "Final f(x):", F1[-1])
print("Final x (alpha=0.5):", X2[-1], "Final f(x):", F2[-1])
print("Final x (alpha=1.2):", X3[-1], "Final f(x):", F3[-1])
plt.plot(X1) # ---- Plot x_k versus iteration ----
plt.xlabel('#iteration'); plt.ylabel('x_k (alpha=0.05)')
plt.show()
plt.plot(X2)
plt.xlabel('#iteration'); plt.ylabel('x_k (alpha=0.5)')
plt.show()
plt.plot(X3)
plt.xlabel('#iteration'); plt.ylabel('x_k (alpha=1.2)')
plt.show()
plt.plot(F1) # ---- Plot f(x_k) versus iteration ----
plt.xlabel('#iteration'); plt.ylabel('f(x_k) (alpha=0.05)')
plt.show()
plt.plot(F2)
plt.xlabel('#iteration'); plt.ylabel('f(x_k) (alpha=0.5)')
plt.show()
plt.plot(F3)
plt.xlabel('#iteration'); plt.ylabel('f(x_k) (alpha=1.2)')
plt.show()

# Q2(i)(a): Contour plot for f(x0, x1) = 0.5(x0^2 + 10 x1^2)
x0, x1 = sympy.symbols('x0, x1', real=True) # Symbolic definition 
x_vec = sympy.Array([x0, x1])
f2 = 0.5*(x_vec[0]**2 + 10*x_vec[1]**2)
grad_f2 = sympy.diff(f2, x_vec)
f2_num = sympy.lambdify((x0, x1), f2) # Numerical contour plot over [-2,2] x [-2,2]
grid_x0 = np.linspace(-2, 2, 400)
grid_x1 = np.linspace(-2, 2, 400)
X0, X1 = np.meshgrid(grid_x0, grid_x1)
F2 = f2_num(X0, X1)
plt.contour(X0, X1, F2, levels=20)
plt.xlabel('x0')
plt.ylabel('x1')
plt.title('Contours of f(x0, x1)')
plt.axis('equal')
plt.show()

# Q2(i)(b): Gradient descent paths for alpha = 0.05 and alpha = 0.2
class QuadraticFn2D():
    def f(self, x):
        return 0.5*(x[0]**2 + 10*(x[1]**2))
    def df(self, x):
        return np.array([x[0], 10*x[1]])
def gradDescent2D(fn, x0, alpha=0.15, num_iters=50):
    x = np.array(x0, dtype=float)
    X = np.array([x])        # store x values
    F = np.array([fn.f(x)])  # store f(x) values
    for _ in range(num_iters):
        step = alpha * fn.df(x)   # step = α∇f(x)
        x = x - step              # x ← x - step
        X = np.append(X, [x], axis=0)
        F = np.append(F, fn.f(x))
    return (X, F)
fn2d = QuadraticFn2D() # Run gradient descent from x^(0) = [1.5, 1.5]
(X_005, F_005) = gradDescent2D(fn2d, x0=[1.5, 1.5], alpha=0.05, num_iters=30)
(X_02,  F_02)  = gradDescent2D(fn2d, x0=[1.5, 1.5], alpha=0.2,  num_iters=30)
grid_x0 = np.linspace(-2, 2, 400) # Contour plot of f over [-2,2] x [-2,2] with overlaid GD paths
grid_x1 = np.linspace(-2, 2, 400)
X0, X1 = np.meshgrid(grid_x0, grid_x1)
F_grid = 0.5*(X0**2 + 10*(X1**2))
plt.contour(X0, X1, F_grid, levels=25)
plt.xlabel('x0')
plt.ylabel('x1')
plt.title('Contours of f(x0, x1) with Gradient Descent Paths')
plt.axis('equal')
plt.plot(X_005[:,0], X_005[:,1], marker='o', markersize=3, label='alpha = 0.05')
plt.plot(X_02[:,0],  X_02[:,1],  marker='o', markersize=3, label='alpha = 0.2')
plt.scatter([1.5], [1.5], s=80, label='Start (1.5, 1.5)')
plt.legend()
plt.show()

# Q2(ii)(a): Plot f(x) = x^4 - 2x^2 + 0.1x for x in [-2,2]
def f_q2(x):
    return x**4 - 2*(x**2) + 0.1*x
xx = np.linspace(-2, 2, 1000)
ff = f_q2(xx)
plt.plot(xx, ff)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot of f(x) = x^4 - 2x^2 + 0.1x over [-2,2]')
plt.show()

# Q2(ii)(b): Gradient descent on f(x)=x^4 - 2x^2 + 0.1x from x0=-1.5 and x0=1.5 with alpha=0.05
class NonconvexFn():
    def f(self, x):
        return x**4 - 2*(x**2) + 0.1*x
    def df(self, x):
        return 4*(x**3) - 4*x + 0.1 # derivative: f'(x) = 4x^3 - 4x + 0.1
def gradDescent1D(fn, x0, alpha=0.05, num_iters=100):
    x = x0
    X = np.array([x])
    F = np.array([fn.f(x)])
    for _ in range(num_iters):
        step = alpha * fn.df(x)
        x = x - step
        X = np.append(X, [x], axis=0)
        F = np.append(F, fn.f(x))
    return (X, F)
fn_nc = NonconvexFn()
(Xm, Fm) = gradDescent1D(fn_nc, x0=-1.5, alpha=0.05, num_iters=100) # Run from both initial points
(Xp, Fp) = gradDescent1D(fn_nc, x0= 1.5, alpha=0.05, num_iters=100)
print("Final x from x0=-1.5:", Xm[-1], "Final f(x):", Fm[-1])
print("Final x from x0= 1.5:", Xp[-1], "Final f(x):", Fp[-1])

# Q3(i)(a): Gradient descent for f(x)=x^2 with different step sizes
class QuadraticFn1D():
    def f(self, x):
        return x**2  # function value f(x)
    def df(self, x):
        return 2*x   # derivative f'(x)
def gradDescent1D(fn, x0, alpha=0.15, num_iters=50):
    x = x0
    X = np.array([x])
    F = np.array([fn.f(x)])
    for _ in range(num_iters):
        step = alpha * fn.df(x)
        x = x - step
        X = np.append(X, [x], axis=0)
        F = np.append(F, fn.f(x))
    return (X, F)
fn_q3 = QuadraticFn1D()
(X_a01,  F_a01)  = gradDescent1D(fn_q3, x0=1, alpha=0.1,  num_iters=50)
(X_a001, F_a001) = gradDescent1D(fn_q3, x0=1, alpha=0.01, num_iters=50)
(X_a101, F_a101) = gradDescent1D(fn_q3, x0=1, alpha=1.01, num_iters=50)

# Q3(i)(b): Plot x_k and f(x_k) vs iteration (including log-scale)
plt.plot(X_a01) # Plot x_k versus iteration
plt.xlabel('#iteration'); plt.ylabel('x_k (alpha=0.1)'); plt.title('Gradient Descent: x_k vs iteration (alpha=0.1)'); plt.show()
plt.plot(X_a001)
plt.xlabel('#iteration'); plt.ylabel('x_k (alpha=0.01)'); plt.title('Gradient Descent: x_k vs iteration (alpha=0.01)'); plt.show()
plt.plot(X_a101) 
plt.xlabel('#iteration'); plt.ylabel('x_k (alpha=1.01)'); plt.title('Gradient Descent: x_k vs iteration (alpha=1.01)'); plt.show()
plt.plot(F_a01) # Plot f(x_k) versus iteration
plt.xlabel('#iteration'); plt.ylabel('f(x_k) (alpha=0.1)'); plt.title('Gradient Descent: f(x_k) vs iteration (alpha=0.1)'); plt.show()
plt.plot(F_a001)
plt.xlabel('#iteration')
plt.ylabel('f(x_k) (alpha=0.01)')
plt.title('Gradient Descent: f(x_k) vs iteration (alpha=0.01)')
plt.show()
plt.plot(F_a101)
plt.xlabel('#iteration')
plt.ylabel('f(x_k) (alpha=1.01)')
plt.title('Gradient Descent: f(x_k) vs iteration (alpha=1.01)')
plt.show()
plt.semilogy(F_a01) # Log-scale plots of f(x_k)
plt.xlabel('#iteration')
plt.ylabel('f(x_k) (log scale)')
plt.title('Gradient Descent: log f(x_k) vs iteration (alpha=0.1)')
plt.show()
plt.semilogy(F_a001)
plt.xlabel('#iteration')
plt.ylabel('f(x_k) (log scale)')
plt.title('Gradient Descent: log f(x_k) vs iteration (alpha=0.01)')
plt.show()
plt.semilogy(F_a101)
plt.xlabel('#iteration')
plt.ylabel('f(x_k) (log scale)')
plt.title('Gradient Descent: log f(x_k) vs iteration (alpha=1.01)')
plt.show()

# Q3(ii)(a): Gradient descent for f(x)=gamma*x^2 with gamma in {0.5,1,2,5}, fix x0=1 and alpha=0.1
class GammaQuadraticFn():
    def __init__(self, gamma):
        self.gamma = gamma
    def f(self, x):
        return self.gamma * (x**2)
    def df(self, x):
        return 2 * self.gamma * x   # f'(x)=2*gamma*x
def gradDescent1D(fn, x0, alpha=0.1, num_iters=50):
    x = x0
    X = np.array([x])
    F = np.array([fn.f(x)])
    for _ in range(num_iters):
        step = alpha * fn.df(x)
        x = x - step
        X = np.append(X, [x], axis=0)
        F = np.append(F, fn.f(x))
    return (X, F)
gammas = [0.5, 1, 2, 5]
results = {}
for g in gammas:
    fn_g = GammaQuadraticFn(g)
    (Xg, Fg) = gradDescent1D(fn_g, x0=1, alpha=0.1, num_iters=50)
    results[g] = (Xg, Fg)

# Q3(ii)(b): Plot f(x_k) vs iteration (log-scale) for each gamma
for g in gammas:
    (Xg, Fg) = results[g]
    plt.semilogy(Fg)
    plt.xlabel('#iteration')
    plt.ylabel('f(x_k) (log scale)')
    plt.title('Gradient Descent: log f(x_k) vs iteration (gamma=' + str(g) + ')')
    plt.show()

# Q3(iii)(a): Subgradient descent for f(x)=|x|, g(x)=sign(x), g(0)=0, Run with x0=1, alpha=0.1, 60 iterations
class AbsFn():
    def f(self, x):
        return abs(x)
    def df(self, x):
        if x > 0: # subgradient g(x) = sign(x), with g(0)=0
            return 1.0
        elif x < 0:
            return -1.0
        else:
            return 0.0
def gradDescent1D(fn, x0, alpha=0.1, num_iters=60):
    x = x0
    X = np.array([x])
    F = np.array([fn.f(x)])
    for _ in range(num_iters):
        step = alpha * fn.df(x)
        x = x - step
        X = np.append(X, [x], axis=0)
        F = np.append(F, fn.f(x))
    return (X, F)
fn_abs = AbsFn()
(X_abs, F_abs) = gradDescent1D(fn_abs, x0=1, alpha=0.1, num_iters=60)

# Q3(iii)(b): Plot x_k and f(x_k) vs iteration
plt.plot(X_abs)
plt.xlabel('#iteration')
plt.ylabel('x_k')
plt.title('Subgradient Descent: x_k vs iteration for f(x)=|x| (alpha=0.1)')
plt.show()
plt.plot(F_abs)
plt.xlabel('#iteration')
plt.ylabel('f(x_k)')
plt.title('Subgradient Descent: f(x_k) vs iteration for f(x)=|x| (alpha=0.1)')
plt.show()

# Q4(a): Contour plots for f(x1,x2)=x1^2 + gamma*x2^2 for gamma = 1 and gamma = 4 over [-1.5,1.5]^2
def f_q4(X1, X2, gamma):
    return X1**2 + gamma*(X2**2)
grid_x1 = np.linspace(-1.5, 1.5, 400)
grid_x2 = np.linspace(-1.5, 1.5, 400)
X1, X2 = np.meshgrid(grid_x1, grid_x2)
F1 = f_q4(X1, X2, gamma=1)
plt.contour(X1, X2, F1, levels=20)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Contours of f(x1,x2) = x1^2 + 1*x2^2 (gamma=1)')
plt.axis('equal')
plt.show()
F4 = f_q4(X1, X2, gamma=4)
plt.contour(X1, X2, F4, levels=20)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Contours of f(x1,x2) = x1^2 + 4*x2^2 (gamma=4)')
plt.axis('equal')
plt.show()

# Q4(b): Gradient descent paths from x^(0)=[1,1] with alpha=0.1. Overlay paths on contours for gamma=1 and gamma=4
class Q4Fn2D():
    def __init__(self, gamma):
        self.gamma = gamma
    def f(self, x):
        return x[0]**2 + self.gamma*(x[1]**2)
    def df(self, x):
        return np.array([2*x[0], 2*self.gamma*x[1]])
def gradDescent2D(fn, x0, alpha=0.1, num_iters=30):
    x = np.array(x0, dtype=float)
    X = np.array([x])
    F = np.array([fn.f(x)])
    for _ in range(num_iters):
        step = alpha * fn.df(x)
        x = x - step
        X = np.append(X, [x], axis=0)
        F = np.append(F, fn.f(x))
    return (X, F)
grid_x1 = np.linspace(-1.5, 1.5, 400) # ---- grid for contour plots ----
grid_x2 = np.linspace(-1.5, 1.5, 400)
X1, X2 = np.meshgrid(grid_x1, grid_x2)
fn_g1 = Q4Fn2D(gamma=1) # ---- run for gamma = 1 ----
(X_g1, F_g1) = gradDescent2D(fn_g1, x0=[1, 1], alpha=0.1, num_iters=30)
F_grid_g1 = X1**2 + 1*(X2**2)
plt.contour(X1, X2, F_grid_g1, levels=25)
plt.plot(X_g1[:,0], X_g1[:,1], marker='o', markersize=3, label='GD path (gamma=1)')
plt.scatter([1], [1], s=80, label='Start (1,1)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Gradient Descent Path on Contours (gamma=1, alpha=0.1)')
plt.axis('equal')
plt.legend()
plt.show()
fn_g4 = Q4Fn2D(gamma=4) # ---- run for gamma = 4 ----
(X_g4, F_g4) = gradDescent2D(fn_g4, x0=[1, 1], alpha=0.1, num_iters=30)
F_grid_g4 = X1**2 + 4*(X2**2)
plt.contour(X1, X2, F_grid_g4, levels=25)
plt.plot(X_g4[:,0], X_g4[:,1], marker='o', markersize=3, label='GD path (gamma=4)')
plt.scatter([1], [1], s=80, label='Start (1,1)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Gradient Descent Path on Contours (gamma=4, alpha=0.1)')
plt.axis('equal')
plt.legend()
plt.show()

# Q4(c): Contour plot for f(x1,x2) = (1-x1)^2 + 100(x2-x1^2)^2 over [-2,2] x [-1,3]
def rosenbrock(X1, X2):
    return (1 - X1)**2 + 100*(X2 - X1**2)**2
grid_x1 = np.linspace(-2, 2, 600)
grid_x2 = np.linspace(-1, 3, 600)
X1, X2 = np.meshgrid(grid_x1, grid_x2)
F_rosen = rosenbrock(X1, X2)
plt.contour(X1, X2, F_rosen, levels=30)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Contours of f(x1,x2) over [-2,2] x [-1,3]')
plt.axis('equal')
plt.show()

# Q4(d): Gradient descent from x^(0)=[-1.25,0.5], alpha = 0.001 and 0.005, 2000 iterations
class RosenbrockFn():
    def f(self, x):
        x1, x2 = x[0], x[1]
        return (1 - x1)**2 + 100*(x2 - x1**2)**2
    def df(self, x):
        x1, x2 = x[0], x[1]         # Gradient: d/dx1 = -2(1-x1) - 400*x1*(x2 - x1^2), d/dx2 = 200*(x2 - x1^2)
        return np.array([
            -2*(1 - x1) - 400*x1*(x2 - x1**2),
            200*(x2 - x1**2)
        ])
def gradDescent2D(fn, x0, alpha=0.001, num_iters=2000):
    x = np.array(x0, dtype=float)
    X = np.array([x])
    F = np.array([fn.f(x)])
    for _ in range(num_iters):
        step = alpha * fn.df(x)
        x = x - step
        X = np.append(X, [x], axis=0)
        F = np.append(F, fn.f(x))
    return (X, F)
fn_rosen = RosenbrockFn()
(X_a001, F_a001) = gradDescent2D(fn_rosen, x0=[-1.25, 0.5], alpha=0.001, num_iters=2000)
(X_a005, F_a005) = gradDescent2D(fn_rosen, x0=[-1.25, 0.5], alpha=0.005, num_iters=2000)

print("Final (alpha=0.001): x =", X_a001[-1], "f(x) =", F_a001[-1])
print("Final (alpha=0.005): x =", X_a005[-1], "f(x) =", F_a005[-1])