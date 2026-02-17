import numpy as np

class QuadraticFn: # Example function from the lecture: f(x)=0.5(x1^2 + 10 x2^2)
    def f(self, x):
        return 0.5 * (x[0]**2 + 10 * x[1]**2) # x is a numpy array [x1, x2]
    
    def df(self, x):
        return np.array([x[0], 10*x[1]]) # gradient of f: # df/dx1 = x1, # df/dx2 = 10*x2
    
fn = QuadraticFn()
x = np.array([1.5, 1.5]) # define x (starting point)

alpha = 1
beta = 0.5; c = 0.5  # design parameters
df = fn.df(x)
while fn.f(x - alpha*df) > fn.f(x) - c*alpha*np.dot(df, df):
    alpha = beta*alpha
    
print("alpha found =", alpha)
print("old x =", x)
print("new x =", x - alpha * df)
