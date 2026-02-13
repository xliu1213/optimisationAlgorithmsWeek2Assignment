import numpy as np
import matplotlib.pyplot as plt

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

fn = QuarticFn()

(X1, F1) = gradDescent(fn, x0=1, alpha=0.05)
(X2, F2) = gradDescent(fn, x0=1, alpha=0.5)
(X3, F3) = gradDescent(fn, x0=1, alpha=1.2)

print("Final x (alpha=0.05):", X1[-1], "Final f(x):", F1[-1])
print("Final x (alpha=0.5):", X2[-1], "Final f(x):", F2[-1])
print("Final x (alpha=1.2):", X3[-1], "Final f(x):", F3[-1])

# ---- Plot x_k versus iteration ----
plt.plot(X1)
plt.xlabel('#iteration'); plt.ylabel('x_k (alpha=0.05)')
plt.show()

plt.plot(X2)
plt.xlabel('#iteration'); plt.ylabel('x_k (alpha=0.5)')
plt.show()

plt.plot(X3)
plt.xlabel('#iteration'); plt.ylabel('x_k (alpha=1.2)')
plt.show()

# ---- Plot f(x_k) versus iteration ----
plt.plot(F1)
plt.xlabel('#iteration'); plt.ylabel('f(x_k) (alpha=0.05)')
plt.show()

plt.plot(F2)
plt.xlabel('#iteration'); plt.ylabel('f(x_k) (alpha=0.5)')
plt.show()

plt.plot(F3)
plt.xlabel('#iteration'); plt.ylabel('f(x_k) (alpha=1.2)')
plt.show()