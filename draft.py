import numpy as np

def f(x):
    x1, x2 = x
    return (x1 - 1.2)**2 + 2 * (x2 - 2.5)**2 + 0.4 * x1 * x2

def grad_f(x):
    x1, x2 = x
    df_dx1 = 2 * (x1 - 1.2) + 0.4 * x2
    df_dx2 = 4 * (x2 - 2.5) + 0.4 * x1
    return np.array([df_dx1, df_dx2], dtype=float)

def project_x1_ge_05(z):
    x1, x2 = z
    return np.array([max(x1, 0.5), x2], dtype=float)

def project_x2_ge_05(z):
    x1, x2 = z
    return np.array([x1, max(x2, 0.5)], dtype=float)

def project_x1_le_x2(z):
    x1, x2 = z
    if x1 <= x2:
        return np.array([x1, x2], dtype=float)
    avg = 0.5 * (x1 + x2)
    return np.array([avg, avg], dtype=float)

def project_X2(z, num_repeats=10):
    x = np.array(z, dtype=float)
    for _ in range(num_repeats):
        x = project_x1_ge_05(x)
        x = project_x2_ge_05(x)
        x = project_x1_le_x2(x)
    return x

def projected_gradient_descent(f, grad_f, project, x0, alpha=0.1, num_iters=60):
    x = np.array(x0, dtype=float)
    X = [x.copy()]
    F = [f(x)]
    for _ in range(num_iters):
        step = alpha * grad_f(x)
        z = x - step
        x = project(z)
        X.append(x.copy())
        F.append(f(x))
    return np.array(X), np.array(F)

x0 = np.array([2.4, 0.7], dtype=float)

X, F = projected_gradient_descent(
    f=f,
    grad_f=grad_f,
    project=project_X2,
    x0=x0,
    alpha=0.1,
    num_iters=60
)