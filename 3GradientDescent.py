import numpy as np
import matplotlib.pyplot as plt

class QuadraticFn():
    def f(self, x):
        return x**2 # function value f(x)

    def df(self, x):
        return 2*x # derivative of f(x)

def gradDescent(fn, x0, alpha=0.15, num_iters=50):
    x = x0;
    X = np.array([x]); F = np.array(fn.f(x));
    for _ in range(num_iters):
        step = alpha*fn.df(x)
        x = x - step
        X = np.append(X, [x], axis=0); F = np.append(F, fn.f(x))
    return (X, F)

fn = QuadraticFn()
(X, F) = gradDescent(fn, x0=1, alpha=0.1)

print("First 4 x values:", X[:4])
print("First 4 f(x) values:", F[:4])

xx = np.arange(-1, 1.1, 0.1)
plt.plot(xx, fn.f(xx))
plt.xlabel('x'); plt.ylabel('f(x)')
plt.show()
plt.plot(F)
plt.xlabel('#iteration'); plt.ylabel('function value')
plt.show()
plt.semilogy(F)
plt.xlabel('#iteration'); plt.ylabel('function value')
plt.show()
plt.plot(X)
plt.xlabel('#iteration'); plt.ylabel('x')
plt.show()
plt.step(X, fn.f(X))
xx = np.arange(-1, 1.1, 0.1)
plt.plot(xx, fn.f(xx))
plt.xlabel('x'); plt.ylabel('f(x)')
plt.show()



