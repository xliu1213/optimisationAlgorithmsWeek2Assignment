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

# Q2 (b)
def f_q2(x): # Objective function for Q2
    x1, x2 = x
    return (x1 - 0.2)**2 + (x2 - 2.0)**2

def g1(x): # Constraint g1(x) <= 0  <=>  x1 >= 0.5
    x1, _ = x
    return 0.5 - x1

def g2(x): # Constraint g2(x) <= 0  <=>  x1*x2 >= 1
    x1, x2 = x
    return 1.0 - x1 * x2

def F_q2(x, lam1, lam2): # Penalised objective
    return f_q2(x) + lam1 * max(0.0, g1(x)) + lam2 * max(0.0, g2(x))

def grad_F_q2(x, lam1, lam2): # Gradient of penalised objective
    x1, x2 = x
    grad = np.array([2 * (x1 - 0.2), 2 * (x2 - 2.0)], dtype=float) # Gradient of original objective f
    if g1(x) > 0: # Add penalty gradient for g1 if active
        grad += np.array([-lam1, 0.0], dtype=float)
    if g2(x) > 0: # Add penalty gradient for g2 if active
        grad += np.array([-lam2 * x2, -lam2 * x1], dtype=float)
    return grad

def gradient_descent_penalty(f_pen, grad_pen, x0, lam1, lam2, alpha=0.05, num_iters=100):
    x = np.array(x0, dtype=float)
    X = [x.copy()]
    F_vals = [f_pen(x, lam1, lam2)]
    for _ in range(num_iters):
        x = x - alpha * grad_pen(x, lam1, lam2)
        X.append(x.copy())
        F_vals.append(f_pen(x, lam1, lam2))
    return np.array(X), np.array(F_vals)

x0_q2 = np.array([1.4, 0.6], dtype=float)
X_small, F_small = gradient_descent_penalty(f_pen=F_q2, grad_pen=grad_F_q2, x0=x0_q2, lam1=0.5, lam2=0.5, alpha=0.05, num_iters=100) # Small penalty weights
X_large, F_large = gradient_descent_penalty(f_pen=F_q2, grad_pen=grad_F_q2, x0=x0_q2, lam1=4.0, lam2=4.0, alpha=0.05, num_iters=100) # Large penalty weights

# Q2 (c)
def G_q2(x, lam1, lam2): # Primal-dual objective
    return f_q2(x) + lam1 * g1(x) + lam2 * g2(x)

def grad_G_q2(x, lam1, lam2): # Gradient of primal-dual objective
    x1, x2 = x
    return np.array([
        2 * (x1 - 0.2) - lam1 - lam2 * x2,
        2 * (x2 - 2.0) - lam2 * x1
    ], dtype=float)

def primal_dual_q2(x0, alpha=0.06, beta=0.08, num_iters=100, lam0=None):
    x = np.array(x0, dtype=float)
    lam = np.array(lam0, dtype=float)
    X = [x.copy()]
    F_vals = [f_q2(x)]
    G_vals = [np.array([g1(x), g2(x)], dtype=float)]
    Lam_vals = [lam.copy()]
    for _ in range(num_iters):
        x = x - alpha * grad_G_q2(x, lam[0], lam[1])
        lam[0] = max(0.0, lam[0] + beta * g1(x))
        lam[1] = max(0.0, lam[1] + beta * g2(x))
        X.append(x.copy())
        F_vals.append(f_q2(x))
        G_vals.append(np.array([g1(x), g2(x)], dtype=float))
        Lam_vals.append(lam.copy())
    return np.array(X), np.array(F_vals), np.array(G_vals), np.array(Lam_vals)

X_pd, F_pd, G_pd, Lam_pd = primal_dual_q2(x0=x0_q2, alpha=0.06, beta=0.08, num_iters=100, lam0=[0.0, 0.0])

# Q2 (d)
def plot_q2_contour_trajectory(X_path, x0, title=None):
    x1_vals = np.linspace(0.0, 1.8, 400)
    x2_vals = np.linspace(0.0, 2.6, 400)
    X1_grid, X2_grid = np.meshgrid(x1_vals, x2_vals)
    F_grid = (X1_grid - 0.2)**2 + (X2_grid - 2.0)**2
    plt.contour(X1_grid, X2_grid, F_grid, levels=25)
    x1_curve = np.linspace(0.5, 1.8, 400) # Feasible region boundary: x1 >= 0.5  -> vertical boundary x1 = 0.5, x1*x2 >= 1 -> boundary x2 = 1/x1
    x2_curve = 1.0 / x1_curve
    plt.plot([0.5, 0.5], [0.0, 2.6], '--', label=r'$x_1 = 0.5$')
    plt.plot(x1_curve, x2_curve, '--', label=r'$x_1 x_2 = 1$')
    plt.plot(X_path[:, 0], X_path[:, 1], marker='o', markersize=3, label='trajectory')
    plt.plot(x0[0], x0[1], 'o', label='start')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(title)
    plt.legend()
    plt.show()

plot_q2_contour_trajectory(X_small, x0_q2, title='Q2(d): fixed penalty, small weights ($\\lambda_1=\\lambda_2=0.5$)') # (I) Small-penalty contour plot with feasible region and trajectory
plot_q2_contour_trajectory(X_large, x0_q2, title='Q2(d): fixed penalty, large weights ($\\lambda_1=\\lambda_2=4.0$)')  # (II) Large-penalty contour plot with feasible region and trajectory

def constraint_history(X_path):
    G_vals = [np.array([g1(x), g2(x)], dtype=float) for x in X_path]
    return np.array(G_vals)

def plot_constraint_violations(G_vals, title=None):
    plt.plot(G_vals[:, 0], label=r'$g_1(x^{(t)}) = 0.5 - x_1^{(t)}$')
    plt.plot(G_vals[:, 1], label=r'$g_2(x^{(t)}) = 1 - x_1^{(t)} x_2^{(t)}$')
    plt.axhline(0.0, linestyle='--')
    plt.xlabel('#iteration')
    plt.ylabel('constraint value')
    plt.title(title)
    plt.legend()
    plt.show()

G_small = constraint_history(X_small)
G_large = constraint_history(X_large)

plot_constraint_violations(G_small, title='Q2(d): constraint violations, small weights ($\\lambda_1=\\lambda_2=0.5$)') # (III) Constraint violations versus iteration
plot_constraint_violations(G_large, title='Q2(d): constraint violations, large weights ($\\lambda_1=\\lambda_2=4.0$)')

# Q2 (e)
plot_q2_contour_trajectory(X_pd, x0_q2, title='Q2(e): primal-dual trajectory ($\\alpha=0.06$, $\\beta=0.08$)')

def plot_two_series(Y, label1, label2, ylabel='value', title=None):
    plt.plot(Y[:, 0], label=label1)
    plt.plot(Y[:, 1], label=label2)
    plt.xlabel('#iteration')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

plot_two_series(
    X_pd,
    label1=r'$x_1^{(t)}$',
    label2=r'$x_2^{(t)}$',
    ylabel='value',
    title='Q2(e): primal-dual coordinates ($\\alpha=0.06$, $\\beta=0.08$)'
)

plot_two_series(
    Lam_pd,
    label1=r'$\lambda_1^{(t)}$',
    label2=r'$\lambda_2^{(t)}$',
    ylabel='value',
    title='Q2(e): primal-dual multipliers ($\\alpha=0.06$, $\\beta=0.08$)'
)

# Q3 (c), (d)
def f_q3(x): # Objective function for Q3
    x1, x2 = x
    return (x1 - 1.5)**2 + (x2 - 1.2)**2

def grad_f_q3(x): # Gradient of f
    x1, x2 = x
    return np.array([2 * (x1 - 1.5), 2 * (x2 - 1.2)], dtype=float)

def vertices_X_q3(): # Vertices of X = {x1 >= 0.5, x2 >= 0.5, x1 + x2 <= 4, x1 <= 3, x2 <= 3}
    return np.array([
        [0.5, 0.5],
        [3.0, 0.5],
        [3.0, 1.0],
        [1.0, 3.0],
        [0.5, 3.0]
    ], dtype=float)

def linear_minimization_oracle_q3(grad, verts): # z_t in argmin_{x in X} grad^T x
    values = np.array([np.dot(grad, v) for v in verts])
    j = np.argmin(values)
    return verts[j].copy()

def frank_wolfe_q3(f, grad_f, x0, beta=0.8, num_iters=100):
    x = np.array(x0, dtype=float)
    verts = vertices_X_q3()
    X = [x.copy()]
    F = [f(x)]
    Z = []
    for _ in range(num_iters):
        z = linear_minimization_oracle_q3(grad_f(x), verts)
        x = beta * x + (1.0 - beta) * z
        X.append(x.copy())
        Z.append(z.copy())
        F.append(f(x))
    return np.array(X), np.array(F), np.array(Z)

def plot_q3_contour_trajectory(X_path, verts, x0, beta, title=None):
    x1_vals = np.linspace(0.0, 3.4, 400)
    x2_vals = np.linspace(0.0, 3.4, 400)
    X1_grid, X2_grid = np.meshgrid(x1_vals, x2_vals)
    F_grid = (X1_grid - 1.5)**2 + (X2_grid - 1.2)**2

    plt.contour(X1_grid, X2_grid, F_grid, levels=25) # (I) Contour plot

    # Feasible region boundary
    plt.plot([0.5, 3.0, 3.0, 1.0, 0.5, 0.5],
             [0.5, 0.5, 1.0, 3.0, 3.0, 0.5], '--', label='feasible boundary')

    # Vertices
    plt.plot(verts[:, 0], verts[:, 1], 'o', label='vertices')

    # Trajectory
    plt.plot(X_path[:, 0], X_path[:, 1], marker='o', markersize=3, label='trajectory')
    plt.plot(x0[0], x0[1], 'o', label='start')

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(title if title is not None else f'Q3(d): contour + trajectory ($\\beta={beta}$)')
    plt.xlim(0.0, 3.4)
    plt.ylim(0.0, 3.4)
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.show()

def plot_q3_value_history(F_vals, beta, title=None):
    plt.plot(F_vals) # (II) f(x^(t)) versus iteration
    plt.xlabel('#iteration')
    plt.ylabel('$f(x^{(t)})$')
    plt.title(title if title is not None else f'Q3(d): objective history ($\\beta={beta}$)')
    plt.show()

def plot_q3_xz_coordinates(X_path, Z_path, beta, title=None):
    plt.plot(X_path[:, 0], label=r'$x_1^{(t)}$')
    plt.plot(X_path[:, 1], label=r'$x_2^{(t)}$')

    # Z has length num_iters, while X has length num_iters+1
    plt.plot(np.arange(1, len(Z_path) + 1), Z_path[:, 0], '--', label=r'$z_1^{(t)}$')
    plt.plot(np.arange(1, len(Z_path) + 1), Z_path[:, 1], '--', label=r'$z_2^{(t)}$')

    plt.xlabel('#iteration')
    plt.ylabel('value')
    plt.title(title if title is not None else f'Q3(d): $x^{{(t)}}$ and $z^{{(t)}}$ coordinates ($\\beta={beta}$)')
    plt.legend()
    plt.show()

x0_q3 = np.array([2.8, 0.8], dtype=float)
verts_q3 = vertices_X_q3()

# beta = 0.8
X_q3_b08, F_q3_b08, Z_q3_b08 = frank_wolfe_q3(
    f=f_q3,
    grad_f=grad_f_q3,
    x0=x0_q3,
    beta=0.8,
    num_iters=100
)

plot_q3_contour_trajectory(
    X_q3_b08,
    verts_q3,
    x0_q3,
    beta=0.8,
    title='Q3(d): contour + feasible region + vertices + trajectory ($\\beta=0.8$)'
)
plot_q3_value_history(
    F_q3_b08,
    beta=0.8,
    title='Q3(d): objective history ($\\beta=0.8$)'
)
plot_q3_xz_coordinates(
    X_q3_b08,
    Z_q3_b08,
    beta=0.8,
    title='Q3(d): $x^{(t)}$ and $z^{(t)}$ coordinates ($\\beta=0.8$)'
)

# beta = 0.95
X_q3_b095, F_q3_b095, Z_q3_b095 = frank_wolfe_q3(
    f=f_q3,
    grad_f=grad_f_q3,
    x0=x0_q3,
    beta=0.95,
    num_iters=100
)

plot_q3_contour_trajectory(
    X_q3_b095,
    verts_q3,
    x0_q3,
    beta=0.95,
    title='Q3(d): contour + feasible region + vertices + trajectory ($\\beta=0.95$)'
)
plot_q3_value_history(
    F_q3_b095,
    beta=0.95,
    title='Q3(d): objective history ($\\beta=0.95$)'
)
plot_q3_xz_coordinates(
    X_q3_b095,
    Z_q3_b095,
    beta=0.95,
    title='Q3(d): $x^{(t)}$ and $z^{(t)}$ coordinates ($\\beta=0.95$)'
)

# Q3 (e)
def project_x1_ge_05_q3(z): # Projection onto {x1 >= 0.5}
    x1, x2 = z
    return np.array([max(x1, 0.5), x2], dtype=float)

def project_x2_ge_05_q3(z): # Projection onto {x2 >= 0.5}
    x1, x2 = z
    return np.array([x1, max(x2, 0.5)], dtype=float)

def project_x1_plus_x2_le_4_q3(z): # Projection onto {x1 + x2 <= 4}
    x1, x2 = z
    if x1 + x2 <= 4.0:
        return np.array([x1, x2], dtype=float)
    shift = 0.5 * (x1 + x2 - 4.0)
    return np.array([x1 - shift, x2 - shift], dtype=float)

def project_x1_le_3_q3(z): # Projection onto {x1 <= 3}
    x1, x2 = z
    return np.array([min(x1, 3.0), x2], dtype=float)

def project_x2_le_3_q3(z): # Projection onto {x2 <= 3}
    x1, x2 = z
    return np.array([x1, min(x2, 3.0)], dtype=float)

def project_X_q3(z, num_repeats=10): # Repeated projections onto X
    x = np.array(z, dtype=float)
    for _ in range(num_repeats):
        x = project_x1_ge_05_q3(x)
        x = project_x2_ge_05_q3(x)
        x = project_x1_plus_x2_le_4_q3(x)
        x = project_x1_le_3_q3(x)
        x = project_x2_le_3_q3(x)
    return x

def projected_gradient_descent_q3(f, grad_f, project, x0, alpha=0.1, num_iters=100):
    x = np.array(x0, dtype=float)
    X = [x.copy()]
    F = [f(x)]
    for _ in range(num_iters):
        z = x - alpha * grad_f(x)
        x = project(z)
        X.append(x.copy())
        F.append(f(x))
    return np.array(X), np.array(F)

x0_q3 = np.array([2.8, 0.8], dtype=float)

X_pgd_q3, F_pgd_q3 = projected_gradient_descent_q3(
    f=f_q3,
    grad_f=grad_f_q3,
    project=project_X_q3,
    x0=x0_q3,
    alpha=0.1,
    num_iters=100
)

print("Q3(e): final PGD iterate =", X_pgd_q3[-1])
print("Q3(e): final PGD objective =", F_pgd_q3[-1])
print("Q3(e): final Frank-Wolfe objective (beta=0.8) =", F_q3_b08[-1])
print("Q3(e): final Frank-Wolfe objective (beta=0.95) =", F_q3_b095[-1])