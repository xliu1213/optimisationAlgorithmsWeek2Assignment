import numpy as np

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
