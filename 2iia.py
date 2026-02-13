import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**4 - 2*(x**2) + 0.1*x

xx = np.linspace(-2, 2, 1000)
ff = f(xx)

plt.plot(xx, ff)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot of f(x) = x^4 - 2x^2 + 0.1x over [-2,2]')
plt.show()
