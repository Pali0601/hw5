import numpy as np
import matplotlib.pyplot as plt

def f(t, y):
    return 1 + (y/t) + (y/t)**2

def exact_solution(t):
    return np.tan(np.log(t))

def euler_method(f, t0, y0, h, t_end):
    t_vals = [t0]
    y_vals = [y0]
    t = t0
    y = y0
    while t < t_end:
        y += h * f(t, y)
        t += h
        t_vals.append(t)
        y_vals.append(y)
    return np.array(t_vals), np.array(y_vals)

# Parameters
t0, y0, h, t_end = 1.0, 0.0, 0.1, 2.0
t_euler, y_euler = euler_method(f, t0, y0, h, t_end)

# Exact values
y_exact = exact_solution(t_euler)

# Display comparison
print("t\tEuler\tExact\tError")
for t, ye, ye_exact in zip(t_euler, y_euler, y_exact):
    print(f"{t:.1f}\t{ye:.6f}\t{ye_exact:.6f}\t{abs(ye - ye_exact):.6f}")
