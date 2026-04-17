import numpy as np
import matplotlib.pyplot as plt

# Q1 (a), (b)
def f(x): # Objective function
    x1, x2 = x
    return (x1 - 1.2)**2 + 2 * (x2 - 2.5)**2 + 0.4 * x1 * x2

def grad_f(x): # Gradient of f
    x1, x2 = x
    df_dx1 = 2 * (x1 - 1.2) + 0.4 * x2
    df_dx2 = 4 * (x2 - 2.5) + 0.4 * x1
    return np.array([df_dx1, df_dx2], dtype=float)

def project_X1(z): # Projection onto X1 = {0.5 <= x1 <= 2.5, 0.5 <= x2 <= 3.5}
    z1, z2 = z
    x1_proj = np.clip(z1, 0.5, 2.5)
    x2_proj = np.clip(z2, 0.5, 3.5)
    return np.array([x1_proj, x2_proj], dtype=float)

def projected_gradient_descent(f, grad_f, project, x0, alpha=0.1, num_iters=60):
    x = np.array(x0, dtype=float)
    X = [x.copy()]
    F = [f(x)]
    for _ in range(num_iters):
        step = alpha * grad_f(x)
        z = x - step              # gradient step
        x = project(z)            # projection step
        X.append(x.copy())
        F.append(f(x))
    return np.array(X), np.array(F)

x0 = np.array([2.4, 0.7], dtype=float)
X, F = projected_gradient_descent(f=f, grad_f=grad_f, project=project_X1, x0=x0, alpha=0.1, num_iters=60)

# Q1 (c)
def project_x1_ge_05(z): # Projection onto {x1 >= 0.5}
    x1, x2 = z
    return np.array([max(x1, 0.5), x2], dtype=float)

def project_x2_ge_05(z): # Projection onto {x2 >= 0.5}
    x1, x2 = z
    return np.array([x1, max(x2, 0.5)], dtype=float)

def project_x1_le_x2(z): # Projection onto {x1 <= x2}
    x1, x2 = z
    if x1 <= x2:
        return np.array([x1, x2], dtype=float)
    avg = 0.5 * (x1 + x2)
    return np.array([avg, avg], dtype=float)

def project_X2(z, num_repeats=10): # Repeated projections onto X2 = {x1 >= 0.5, x2 >= 0.5, x1 <= x2}
    x = np.array(z, dtype=float)
    for _ in range(num_repeats):
        x = project_x1_ge_05(x)
        x = project_x2_ge_05(x)
        x = project_x1_le_x2(x)
    return x

X2, F2 = projected_gradient_descent(f=f, grad_f=grad_f, project=project_X2, x0=x0, alpha=0.1, num_iters=60)

def plot_results(X_path, F_vals, x0, feasible_set='X1'):
    x1_vals = np.linspace(0.0, 3.0, 400) # (I) Contour plot of f(x1,x2) with feasible region and trajectory
    x2_vals = np.linspace(0.0, 4.0, 400)
    X1_grid, X2_grid = np.meshgrid(x1_vals, x2_vals)
    F_grid = (X1_grid - 1.2)**2 + 2 * (X2_grid - 2.5)**2 + 0.4 * X1_grid * X2_grid

    plt.contour(X1_grid, X2_grid, F_grid, levels=25) # (I) Contour plot
    plt.plot(X_path[:, 0], X_path[:, 1], marker='o', markersize=3)
    plt.plot(x0[0], x0[1], 'o')

    if feasible_set == 'X1':
        plt.plot([0.5, 2.5, 2.5, 0.5, 0.5], [0.5, 0.5, 3.5, 3.5, 0.5], '--')
    elif feasible_set == 'X2':
        plt.plot([0.5, 0.5], [0.5, 4.0], '--')
        plt.plot([0.5, 3.0], [0.5, 3.0], '--')

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.xlim(0.0, 3.0)
    plt.ylim(0.0, 4.0)
    plt.gca().set_aspect('equal')
    plt.show()

    plt.plot(F_vals) # (II) f(x^(t)) versus iteration
    plt.xlabel('#iteration')
    plt.ylabel('$f(x^{(t)})$')
    plt.show()

    plt.plot(X_path[:, 0], label='$x_1^{(t)}$') # (III) x_1^(t) and x_2^(t) versus iteration
    plt.plot(X_path[:, 1], label='$x_2^{(t)}$')
    plt.xlabel('#iteration')
    plt.ylabel('value')
    plt.legend()
    plt.show()

# Q1 (d)
plot_results(X, F, x0, feasible_set='X1')

# Q1 (e)
plot_results(X2, F2, x0, feasible_set='X2')
