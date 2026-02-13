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


