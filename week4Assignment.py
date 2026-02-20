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

