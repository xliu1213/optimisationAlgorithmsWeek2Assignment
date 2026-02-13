import sympy
import numpy as np
import matplotlib.pyplot as plt

# ----- Symbolic definition -----
x0, x1 = sympy.symbols('x0, x1', real=True)
x = sympy.Array([x0, x1])
f = 0.5*(x[0]**2 + 10*x[1]**2)
dfdc = sympy.diff(f, x)
print("f =", f)
print("grad f =", dfdc)
# ----- Numerical contour plot over [-2,2] x [-2,2] -----
f_num = sympy.lambdify((x0, x1), f)
xx0 = np.linspace(-2, 2, 400)
xx1 = np.linspace(-2, 2, 400)
X0, X1 = np.meshgrid(xx0, xx1)
F = f_num(X0, X1)
plt.contour(X0, X1, F, levels=20)
plt.xlabel('x0')
plt.ylabel('x1')
plt.title('Contours of f(x0, x1)')
plt.axis('equal')
plt.show()
