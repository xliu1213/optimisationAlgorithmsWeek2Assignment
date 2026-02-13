import numpy as np
import matplotlib.pyplot as plt

class QuadraticFn2D():
    def f(self, x):
        # x is a vector [x0, x1]
        return 0.5*(x[0]**2 + 10*(x[1]**2))

    def df(self, x):
        # gradient vector
        return np.array([x[0], 10*x[1]])

def gradDescent(fn, x0, alpha=0.15, num_iters=50):
    x = np.array(x0, dtype=float)

    X = np.array([x])       # store x values
    F = np.array([fn.f(x)]) # store f(x) values

    for _ in range(num_iters):
        step = alpha * fn.df(x)   # step = α∇f(x)
        x = x - step              # x ← x - step

        X = np.append(X, [x], axis=0)
        F = np.append(F, fn.f(x))

    return (X, F)

# ------------------------------------------------------------
# Run gradient descent from x^(0) = [1.5, 1.5]
# ------------------------------------------------------------

fn = QuadraticFn2D()

(X_005, F_005) = gradDescent(fn, x0=[1.5, 1.5], alpha=0.05, num_iters=30)
(X_02,  F_02)  = gradDescent(fn, x0=[1.5, 1.5], alpha=0.2,  num_iters=30)

grid_x0 = np.linspace(-2, 2, 400) # Contour plot of f over [-2,2] x [-2,2]
grid_x1 = np.linspace(-2, 2, 400)
X0, X1 = np.meshgrid(grid_x0, grid_x1)
F_grid = 0.5*(X0**2 + 10*(X1**2))
plt.contour(X0, X1, F_grid, levels=25)
plt.xlabel('x0')
plt.ylabel('x1')
plt.title('Contours of f(x0, x1) with Gradient Descent Paths')
plt.axis('equal')
plt.plot(X_005[:,0], X_005[:,1], marker='o', markersize=3, label='alpha = 0.05') # Overlay paths
plt.plot(X_02[:,0],  X_02[:,1],  marker='o', markersize=3, label='alpha = 0.2')
plt.scatter([1.5], [1.5], s=80, label='Start (1.5, 1.5)') # Start point
plt.legend()
plt.show()

