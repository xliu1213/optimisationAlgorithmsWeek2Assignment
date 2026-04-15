import numpy as np

def f(x): # Objective function
    x1, x2 = x
    return (x1 - 1.2)**2 + 2 * (x2 - 2.5)**2 + 0.4 * x1 * x2

def grad_f(x): # Gradient of f
    x1, x2 = x
    df_dx1 = 2 * (x1 - 1.2) + 0.4 * x2
    df_dx2 = 4 * (x2 - 2.5) + 0.4 * x1
    return np.array([df_dx1, df_dx2], dtype=float)

def vertices_X(): # Feasible set X: rectangle with vertices
    return np.array([ # 0.5 <= x1 <= 2.5, 0.5 <= x2 <= 3.5
        [0.5, 0.5],
        [2.5, 0.5],
        [2.5, 3.5],
        [0.5, 3.5]
    ], dtype=float)

def linear_minimization_oracle(grad, verts): # Linear minimization oracle: z_t in argmin_{x in X} grad_f(x_t)^T x
    values = np.array([np.dot(grad, v) for v in verts]) # For a rectangle/polytope, we can check the vertices
    j = np.argmin(values)
    return verts[j].copy()

def frank_wolfe(f, grad_f, beta=0.8, num_iters=100):
    x = np.array([0.0, 0.0], dtype=float) # initialise x0 = 0, t = 0, 0 < beta < 1
    t = 0
    verts = vertices_X()
    X = [x.copy()]
    F = [f(x)]
    Z = []
    while t < num_iters:
        z = linear_minimization_oracle(grad_f(x), verts)
        x = beta * x + (1.0 - beta) * z
        X.append(x.copy())
        F.append(f(x))
        Z.append(z.copy())
        t = t + 1
    return np.array(X), np.array(F), np.array(Z)

X, F, Z = frank_wolfe(
    f=f,
    grad_f=grad_f,
    beta=0.8,
    num_iters=100
)
