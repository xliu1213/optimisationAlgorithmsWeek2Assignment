import numpy as np
import matplotlib.pyplot as plt

dd = 0.0001
x = np.arange(-0.1, 0.1 + dd, dd)
d = 0.1
df = (np.abs(x + d) - np.abs(x)) / d
plt.figure()
plt.plot(x, np.sign(x), label='sign(x)', linewidth=3)
plt.plot(x, df, '--', label='df', linewidth=3)
plt.legend()
plt.xlabel('x')
plt.ylabel('value')
plt.title('Finite difference approximation of derivative')
plt.grid(True)
ff = np.abs(x[0]) + dd * np.cumsum(df) - d / 2 # function corresponding to finite diff approx
plt.figure()
plt.plot(x, np.abs(x), '--', label='abs(x)', linewidth=3)
plt.plot(x + d / 2, ff, label='ff', linewidth=3)
plt.legend()
plt.xlabel('x')
plt.ylabel('value')
plt.title('Function corresponding to finite difference approximation')
plt.grid(True)
plt.show()