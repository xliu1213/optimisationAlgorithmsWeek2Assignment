# import sympy
# x = sympy.symbols('x', real=True)
# f = x**2
# dfdx = sympy.diff(f, x)
# print(f, dfdx)

# import sympy
# x0, x1 = sympy.symbols('x0, x1', real=True)
# x = sympy.Array([x0, x1])
# f = 0.5*(x[0]**2 + 10*x[1]**2)
# dfdc = sympy.diff(f, x)
# print(f, dfdc)

import sympy
import numpy as np
x = sympy.symbols('x', real=True)
f = x**2
dfdx = sympy.diff(f, x)
f = sympy.lambdify(x, f)
dfdx = sympy.lambdify(x, dfdx)
x = np.array([1.0])
print(dfdx(*x))
delta = 0.01
print(((x+delta)**2 - x**2)/delta)

# x = x0
# for k in range(num_iters):
#     step = calcStep(fn, x)
#     x = x - step

