import sympy 

x = sympy.symbols('x', real=True)
f = x**4

dfdx = sympy.diff(f, x)
print(dfdx)
