import sympy
import numpy as np
import matplotlib.pyplot as plt

x = sympy.symbols('x', real=True) # Exact derivative
f = x**4
dfdx = sympy.diff(f, x)

f = sympy.lambdify(x, f) # Forward finite-difference approximation
dfdx = sympy.lambdify(x, dfdx)

x = np.linspace(-2, 2, 1000) #

delta = 0.01
d_exact = dfdx(x)
d_fd = (f(x + delta) - f(x)) / delta # formula in Gradients slides
print("First 5 exact derivative values:", d_exact[:5]) 
print("First 5 finite-diff values:", d_fd[:5]) 

# Plot both
plt.plot(x, d_exact)
plt.xlabel('x'); plt.ylabel('exact derivative')
plt.show()
plt.plot(x, d_fd)
plt.xlabel('x'); plt.ylabel('forward finite-difference approximation with delta = 0.01')
plt.show()