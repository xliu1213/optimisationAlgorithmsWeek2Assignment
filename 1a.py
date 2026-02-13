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