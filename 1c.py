import sympy
import numpy as np
import matplotlib.pyplot as plt

# Symbolic setup
x = sympy.symbols('x', real=True)
f = x**4
dfdx = sympy.diff(f, x)

# Convert to numerical functions
f = sympy.lambdify(x, f)
dfdx = sympy.lambdify(x, dfdx)

# x grid in [-2, 2]
x = np.linspace(-2, 2, 1000)

# Exact derivative values (fixed, doesn't depend on delta)
d_exact = dfdx(x)

# Delta values in [0.001, 1]
deltas = np.linspace(0.001, 1, 200)

mae = np.zeros(len(deltas)) # 

for i in range(len(deltas)):
    delta = deltas[i]
    d_fd = (f(x + delta) - f(x)) / delta
    mae[i] = np.mean(np.abs(d_fd - d_exact))

print("First 5 delta values:", deltas[:5])
print("First 5 MAE values:", mae[:5])

# Plot MAE vs delta
plt.plot(deltas, mae)
plt.xlabel('delta')
plt.ylabel('MAE')
plt.title('Mean Absolute Error vs delta (forward finite difference)')
plt.show()
