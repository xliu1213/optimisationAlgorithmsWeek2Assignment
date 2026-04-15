import numpy as np

def f(x):  # Original objective
    x1, x2 = x
    return (x1 - 1.2)**2 + 2 * (x2 - 2.5)**2 + 0.4 * x1 * x2

def g(x):  # Constraint function g(x) <= 0
    x1, x2 = x  # Example: x1 * x2 >= 1  <=>  1 - x1*x2 <= 0
    return 1.0 - x1 * x2

def grad_f(x):  # Gradient of f
    x1, x2 = x
    df_dx1 = 2 * (x1 - 1.2) + 0.4 * x2
    df_dx2 = 4 * (x2 - 2.5) + 0.4 * x1
    return np.array([df_dx1, df_dx2], dtype=float)

def grad_g(x):  # Gradient of g
    x1, x2 = x
    dg_dx1 = -x2
    dg_dx2 = -x1
    return np.array([dg_dx1, dg_dx2], dtype=float)

def F_lambda(x, lam):  # Penalized objective
    return f(x) + lam * max(0.0, g(x))

def grad_F_lambda(x, lam):  # Gradient of penalized objective
    if g(x) > 0:
        return grad_f(x) + lam * grad_g(x)
    return grad_f(x)

def multiplier_method(x0, alpha=0.05, beta=1.0, num_iters=100, lam0=0.0, projected=True):
    """
    Reusable multiplier method with two lambda update rules.

    projected=True:
        lambda_{t+1} = [lambda_t + beta * g(x_t)]_+
                     = max(0, lambda_t + beta * g(x_t))

    projected=False:
        lambda_{t+1} = lambda_t + beta * max(0, g(x_t))
    """
    x = np.array(x0, dtype=float)
    lam = float(lam0)
    X = [x.copy()]
    F_vals = [f(x)]
    G_vals = [g(x)]
    Lam_vals = [lam]
    Penalized_vals = [F_lambda(x, lam)]
    for _ in range(num_iters):
        x = x - alpha * grad_F_lambda(x, lam) # x-update
        if projected: # lambda-update
            lam = max(0.0, lam + beta * g(x))
        else:
            lam = lam + beta * max(0.0, g(x))
        X.append(x.copy())
        F_vals.append(f(x))
        G_vals.append(g(x))
        Lam_vals.append(lam)
        Penalized_vals.append(F_lambda(x, lam))
    return (
        np.array(X),
        np.array(F_vals),
        np.array(G_vals),
        np.array(Lam_vals),
        np.array(Penalized_vals),
    )
x0 = np.array([2.4, 0.7], dtype=float)

X, F_vals, G_vals, Lam_vals, Penalized_vals = multiplier_method( # Use projected=True for: lambda_{t+1} = [lambda_t + beta g(x_t)]_+
    x0=x0,
    alpha=0.05,
    beta=1.0,
    num_iters=100,
    lam0=0.0,
    projected=True # If you want the earlier update rule instead, use projected=False
)

def G_lambda(x, lam):  # Lagrangian / primal-dual objective
    return f(x) + lam * g(x)

def grad_G_lambda(x, lam):  # Gradient of Lagrangian / primal-dual objective
    return grad_f(x) + lam * grad_g(x)

def primal_dual_method(x0, alpha=0.05, beta=1.0, num_iters=100, lam0=0.0):
    x = np.array(x0, dtype=float)
    lam = float(lam0)
    X = [x.copy()]
    F_vals = [f(x)]
    G_vals = [g(x)]
    Lam_vals = [lam]
    Lagrangian_vals = [G_lambda(x, lam)]
    for _ in range(num_iters):
        x = x - alpha * grad_G_lambda(x, lam)   # x-update
        lam = max(0.0, lam + beta * g(x))       # lambda-update
        X.append(x.copy())
        F_vals.append(f(x))
        G_vals.append(g(x))
        Lam_vals.append(lam)
        Lagrangian_vals.append(G_lambda(x, lam))
    return (
        np.array(X),
        np.array(F_vals),
        np.array(G_vals),
        np.array(Lam_vals),
        np.array(Lagrangian_vals),
    )

X_pd, F_vals_pd, G_vals_pd, Lam_vals_pd, Lagrangian_vals_pd = primal_dual_method( # Example run of the added primal-dual method
    x0=x0,
    alpha=0.05,
    beta=1.0,
    num_iters=100,
    lam0=0.0
)

def g_list(x):  # Multiple constraint functions [g1(x), g2(x), ...]
    x1, x2 = x
    return np.array([
        1.0 - x1 * x2,   # g1(x) <= 0
        0.5 - x1         # g2(x) <= 0
    ], dtype=float)

def grad_g_list(x):  # Gradients of the constraints
    x1, x2 = x
    return [
        np.array([-x2, -x1], dtype=float),  # grad g1
        np.array([-1.0, 0.0], dtype=float)  # grad g2
    ]

def G_lambda_multi(x, lam):  # Multi-constraint Lagrangian
    gvals = g_list(x)
    return f(x) + np.dot(lam, gvals)

def grad_G_lambda_multi(x, lam):  # Gradient of multi-constraint Lagrangian
    grad = grad_f(x).copy()
    grads = grad_g_list(x)
    for i in range(len(lam)):
        grad = grad + lam[i] * grads[i]
    return grad

def primal_dual_method_multi(x0, alpha=0.05, beta=1.0, num_iters=100, lam0=None):
    x = np.array(x0, dtype=float)
    if lam0 is None:
        lam = np.zeros(len(g_list(x)), dtype=float)
    else:
        lam = np.array(lam0, dtype=float)
    X = [x.copy()]
    F_vals = [f(x)]
    G_vals = [g_list(x).copy()]
    Lam_vals = [lam.copy()]
    Lagrangian_vals = [G_lambda_multi(x, lam)]
    for _ in range(num_iters):
        x = x - alpha * grad_G_lambda_multi(x, lam)   # x-update
        lam = np.maximum(0.0, lam + beta * g_list(x)) # lambda-update for all constraints
        X.append(x.copy())
        F_vals.append(f(x))
        G_vals.append(g_list(x).copy())
        Lam_vals.append(lam.copy())
        Lagrangian_vals.append(G_lambda_multi(x, lam))
    return (
        np.array(X),
        np.array(F_vals),
        np.array(G_vals),
        np.array(Lam_vals),
        np.array(Lagrangian_vals),
    )

X_multi, F_vals_multi, G_vals_multi, Lam_vals_multi, Lagrangian_vals_multi = primal_dual_method_multi( # Example run of the multi-constraint version
    x0=x0,
    alpha=0.05,
    beta=1.0,
    num_iters=100
)