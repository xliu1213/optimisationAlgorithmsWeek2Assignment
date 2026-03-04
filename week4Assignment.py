import numpy as np
import matplotlib.pyplot as plt

class Q1Fn: # Function: f(x,y)=x^2+100y^2
    def f(self, x):
        return x[0]**2 + 100*(x[1]**2) # x is np.array([x1, x2]) = [x, y]
    def df(self, x):
        return np.array([2*x[0], 200*x[1]]) # gradient = [2x, 200y]
fn = Q1Fn()
x0 = np.array([2.0, 2.0])
num_iters = 200

def run_polyak(fn, x0, num_iters=200, f_star=0.0, eps=1e-3): # Polyak step size, alpha = (f(x)-f*)/(||grad||^2 + eps)
    x = x0.copy()
    F = np.array([fn.f(x)])
    for _ in range(num_iters):
        g = fn.df(x)
        alpha = (fn.f(x) - f_star) / (np.dot(g, g) + eps)
        x = x - alpha * g
        F = np.append(F, fn.f(x))
    return F

def run_rmsprop(fn, x0, num_iters=200, alpha0=0.2, beta=0.9, eps=1e-5): # RMSprop, sum = beta*sum + (1-beta)*g^2, alpha_t = alpha0/(sqrt(sum)+eps)
    x = x0.copy()
    F = np.array([fn.f(x)])
    sum = np.zeros_like(x)
    for _ in range(num_iters):
        g = fn.df(x)
        sum = beta*sum + (1-beta)*(g**2)
        step = alpha0 * g / (np.sqrt(sum) + eps)
        x = x - step
        F = np.append(F, fn.f(x))
    return F

# Heavy Ball (Polyak Momentum), z_{t+1} = beta*z_t + alpha*grad, x_{t+1} = x_t - z_{t+1}
def run_heavy_ball(fn, x0, num_iters=200, alpha=0.01, beta=0.9):
    x = x0.copy()
    F = np.array([fn.f(x)])
    z = np.zeros_like(x)
    for _ in range(num_iters):
        g = fn.df(x)
        z = beta*z + alpha*g
        x = x - z
        F = np.append(F, fn.f(x))
    return F

# Adam, m = beta1*m + (1-beta1)*g, v = beta2*v + (1-beta2)*g^2, bias correction: m_hat = m/(1-beta1^t), v_hat = v/(1-beta2^t), x = x - alpha * m_hat/(sqrt(v_hat)+eps)
def run_adam(fn, x0, num_iters=200, alpha=0.1, beta1=0.9, beta2=0.999, eps=1e-5):
    x = x0.copy()
    F = np.array([fn.f(x)])
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    t = 0
    for _ in range(num_iters):
        t = t + 1
        g = fn.df(x)
        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*(g**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x = x - alpha * m_hat / (np.sqrt(v_hat) + eps)
        F = np.append(F, fn.f(x))
    return F

F_polyak = run_polyak(fn, x0, num_iters=num_iters, f_star=0.0, eps=1e-3) # Run all methods
F_rmsprop = run_rmsprop(fn, x0, num_iters=num_iters, alpha0=0.2, beta=0.9, eps=1e-5)
F_hb = run_heavy_ball(fn, x0, num_iters=num_iters, alpha=0.01, beta=0.9)
F_adam = run_adam(fn, x0, num_iters=num_iters, alpha=0.1, beta1=0.9, beta2=0.999, eps=1e-5)

plt.semilogy(F_polyak, label="Polyak")
plt.semilogy(F_rmsprop, label="RMSprop (a0=0.2, b=0.9)")
plt.semilogy(F_hb, label="Heavy Ball (a=0.01, b=0.9)")
plt.semilogy(F_adam, label="Adam (a=0.1, b1=0.9, b2=0.999)")

plt.xlabel("#iteration")
plt.ylabel("function value")
plt.legend()
plt.show()

# Q1 (II): Heavy Ball
def run_heavy_ball_xcoord(fn, x0, num_iters=200, alpha=0.01, beta=0.9):
    x = x0.copy()
    z = np.zeros_like(x)
    Xcoord = np.array([x[0]])   # store x-coordinate only
    for _ in range(num_iters):
        g = fn.df(x)
        z = beta*z + alpha*g
        x = x - z
        Xcoord = np.append(Xcoord, x[0])
    return Xcoord
alphas = [0.006, 0.01, 0.02] # given step sizes
plt.figure()
for a in alphas:
    X = run_heavy_ball_xcoord(fn, x0, num_iters=num_iters, alpha=a, beta=0.9)
    plt.plot(X, label=f"alpha = {a}")
plt.xlabel("#iteration")
plt.ylabel("x-coordinate")
plt.title("Heavy Ball Stability Test (β = 0.9)")
plt.legend()
plt.show()

# Q1 (III)
def traj_heavy_ball(fn, x0, num_iters=200, alpha=0.01, beta=0.9): # Return full trajectory X (shape: (num_iters+1, 2))
    x = x0.copy()
    z = np.zeros_like(x)
    X = [x.copy()]
    for _ in range(num_iters):
        g = fn.df(x)
        z = beta*z + alpha*g
        x = x - z
        X.append(x.copy())
    return np.array(X)

def traj_rmsprop(fn, x0, num_iters=200, alpha0=0.2, beta=0.9, eps=1e-5):
    x = x0.copy()
    s = np.zeros_like(x)
    X = [x.copy()]
    for _ in range(num_iters):
        g = fn.df(x)
        s = beta*s + (1-beta)*(g**2)
        step = alpha0 * g / (np.sqrt(s) + eps)
        x = x - step
        X.append(x.copy())
    return np.array(X)

def traj_adam(fn, x0, num_iters=200, alpha=0.1, beta1=0.9, beta2=0.999, eps=1e-5):
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    X = [x.copy()]
    t = 0
    for _ in range(num_iters):
        t += 1
        g = fn.df(x)
        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*(g**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x = x - alpha * m_hat / (np.sqrt(v_hat) + eps)
        X.append(x.copy())
    return np.array(X)

X_hb  = traj_heavy_ball(fn, x0, num_iters=200, alpha=0.01, beta=0.9) # Generate trajectories 
X_rms = traj_rmsprop(fn, x0, num_iters=200, alpha0=0.2, beta=0.9, eps=1e-5)
X_ad  = traj_adam(fn, x0, num_iters=200, alpha=0.1, beta1=0.9, beta2=0.999, eps=1e-5)

grid_x = np.linspace(-2.5, 2.5, 400) # Contours 
grid_y = np.linspace(-2.5, 2.5, 400)
X, Y = np.meshgrid(grid_x, grid_y)
F = fn.f([X, Y]) # Reuse existing function
plt.contour(X, Y, F, levels=30)
plt.plot(X_hb[:,0], X_hb[:,1], linestyle='dashdot', linewidth=2, label="Heavy Ball (β=0.9, α=0.01)") # Heavy Ball 
plt.plot(X_rms[:,0], X_rms[:,1], linestyle='-', linewidth=2, label="RMSprop (β=0.9, α=0.2)") # RMSprop 
plt.plot(X_ad[:,0], X_ad[:,1], linestyle='dotted', markersize=5, label="Adam (β1=0.9, β2=0.999, α=0.1)") # Adam 
plt.scatter(x0[0], x0[1], color='red', s=100, zorder=6, label="start (2,2)")
plt.scatter(0, 0, color='yellow', edgecolor='black', marker='*', s=180, zorder=6, label="minimum")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Q1(III): Contours with optimisation trajectories')
plt.axis('equal')
plt.legend()
plt.show()

# Q2 (I): Rosenbrock: f(x,y) = (1-x)^2 + 100 (y - x^2)^2, x0 = (-1.25, 0.5), run 3000 iterations
class Q2Fn:  
    def f(self, x):
        return (1 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2 # x = np.array([x, y])
    def df(self, x):
        dx = -2.0*(1.0 - x[0]) - 400.0*x[0]*(x[1] - x[0]**2) # grad = [df/dx, df/dy], # df/dx = -2(1-x) - 400x(y-x^2)
        dy = 200.0*(x[1] - x[0]**2) # df/dy = 200(y-x^2)
        return np.array([dx, dy])

fn2 = Q2Fn()
x0_2 = np.array([-1.25, 0.5])
num_iters2 = 3000

def run_polyak_capped(fn, x0, num_iters=3000, f_star=0.0, eps=1e-3, alpha_max=0.1):
    x = x0.copy() # Polyak step size with cap (required: alpha <= 0.1)
    F = np.array([fn.f(x)]) # alpha = (f(x)-f*)/(||grad||^2 + eps), then alpha = min(alpha, alpha_max)
    for _ in range(num_iters):
        g = fn.df(x)
        alpha = (fn.f(x) - f_star) / (np.dot(g, g) + eps)
        alpha = min(alpha, alpha_max)  # enforce α ≤ 0.1
        x = x - alpha * g
        F = np.append(F, fn.f(x))
    return F

F2_polyak = run_polyak_capped(fn2, x0_2, num_iters=num_iters2, f_star=0.0, eps=1e-3, alpha_max=0.1)
F2_rmsprop = run_rmsprop(fn2, x0_2, num_iters=num_iters2, alpha0=0.01, beta=0.9, eps=1e-5)
F2_hb = run_heavy_ball(fn2, x0_2, num_iters=num_iters2, alpha=2e-4, beta=0.9)
F2_adam = run_adam(fn2, x0_2, num_iters=num_iters2, alpha=0.05, beta1=0.9, beta2=0.999, eps=1e-5)

plt.semilogy(F2_polyak, label="Polyak (cap α≤0.1)")
plt.semilogy(F2_rmsprop, label="RMSprop (α=0.01, β=0.9)")
plt.semilogy(F2_hb, label="Heavy Ball (α=2e-4, β=0.9)")
plt.semilogy(F2_adam, label="Adam (α=0.05, β1=0.9, β2=0.999)")
plt.xlabel("#iteration")
plt.ylabel("function value")
plt.title("Q2(I): Rosenbrock — function value vs iteration")
plt.legend()
plt.show()

# Q2 (II): Adam stability test on Rosenbrock, fix β1 = 0.9, β2 = 0.999, increase α until instability occurs
def run_adam_xcoord(fn, x0, num_iters=3000, alpha=0.05, beta1=0.9, beta2=0.999, eps=1e-5):
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    Xcoord = np.array([x[0]])   # store x-coordinate only
    t = 0
    for _ in range(num_iters):
        t += 1
        g = fn.df(x)
        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*(g**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x = x - alpha * m_hat / (np.sqrt(v_hat) + eps)
        Xcoord = np.append(Xcoord, x[0])
    return Xcoord
alphas_q2 = [0.02, 0.05, 0.12] # Given step sizes
plt.figure()
for a in alphas_q2:
    X = run_adam_xcoord(fn2, x0_2, num_iters=num_iters2, alpha=a, beta1=0.9, beta2=0.999)
    plt.plot(X, label=f"alpha = {a}")
plt.xlabel("#iteration")
plt.ylabel("x-coordinate")
plt.title("Q2(II): Adam Stability Test (β1=0.9, β2=0.999)")
plt.legend()
plt.show()

# Q2 (III): Rosenbrock trajectories on contour plot
X2_hb  = traj_heavy_ball(fn2, x0_2, num_iters=num_iters2, alpha=2e-4, beta=0.9) # Generate optimisation trajectories using SAME trajectory functions
X2_rms = traj_rmsprop(fn2, x0_2, num_iters=num_iters2, alpha0=0.01, beta=0.9, eps=1e-5)
X2_ad  = traj_adam(fn2, x0_2, num_iters=num_iters2, alpha=0.05, beta1=0.9, beta2=0.999, eps=1e-5)
grid_x = np.linspace(-2, 2, 400) # Rosenbrock contour grid
grid_y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(grid_x, grid_y)
F = fn2.f([X, Y])   # reuse function 
plt.contour(X, Y, F, levels=30)
plt.plot(X2_hb[:,0], X2_hb[:,1], linestyle='dashdot', linewidth=8, label="Heavy Ball (β=0.9, α=2e-4)") # Overlay trajectories 
plt.plot(X2_rms[:,0], X2_rms[:,1], linestyle='-', linewidth=2, label="RMSprop (β=0.9, α=0.01)")
plt.plot(X2_ad[:,0], X2_ad[:,1], linestyle='dotted', linewidth=2, label="Adam (β1=0.9, β2=0.999, α=0.05)")
plt.scatter(x0_2[0], x0_2[1], color='red', s=100, label="start (-1.25, 0.5)") # start + minimum
plt.scatter(1, 1, color='yellow', edgecolor='black', marker='*', s=200, label="minimum (1,1)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Q2(III): Rosenbrock Contours with Optimisation Trajectories")
plt.axis("equal")
plt.legend()
plt.show()
