import numpy as np 

def f(x): # Objective function
    x1, x2 = x
    return x1**2 + 2 * x2**2

# Pure random search
def pure_random_search(f, x0, num_iters=100, npoints=20, step_size=0.1):  
    x = np.array(x0, dtype=float)
    X = [x.copy()]
    F = [f(x)]
    ndim = len(x)
    for _ in range(num_iters):
        z = np.random.randn(ndim, npoints) # Generate random directions (each column is one direction)
        delta = z / np.sqrt(np.sum(z * z, axis=0))
        y = x[:, None] + step_size * delta # Candidate points around current x
        fy = np.array([f(y[:, j]) for j in range(npoints)]) # Evaluate all candidate points
        j_best = np.argmin(fy) # Pick the best candidate
        y_best = y[:, j_best]
        if f(y_best) < f(x): # Move only if it improves the function value
            x = y_best
        X.append(x.copy())
        F.append(f(x))
    return np.array(X), np.array(F)

np.random.seed(0) # Main
x0 = np.array([1.5, 1.0])
X, F = pure_random_search(
    f,
    x0=x0,
    num_iters=100,
    npoints=30,
    step_size=0.15
)

# Grid search
best_x = None
best_f = float('inf')
for x0 in np.arange(-1, 1, 0.1):
    for x1 in np.arange(-1, 1, 0.1):
        x = np.array([x0, x1], dtype=float)
        fx = f(x)
        if fx < best_f:
            best_f = fx
            best_x = x.copy()

# Uniform random search (global)
best_x = None
best_f = float('inf')
for k in range(1000):  # number of random samples
    x0_rand = np.random.uniform(-1, 1)
    x1_rand = np.random.uniform(-1, 1)
    x = np.array([x0_rand, x1_rand])
    fx = f(x)
    if fx < best_f:
        best_f = fx
        best_x = x.copy()
        