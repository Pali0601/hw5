import numpy as np

def f(t, y):
    return 1 + (y/t) + (y/t)**2

def exact_solution(t):
    return np.tan(np.log(t))

def df_dt(t, y):
    return -y / t**2 - 2 * y**2 / t**3

def df_dy(t, y):
    return 1 / t + 2 * y / t**2

def taylor_method_order2(f, df_dt, df_dy, t0, y0, h, t_end):
    t_vals = [t0]
    y_vals = [y0]
    t = t0
    y = y0
    while t < t_end:
        f_val = f(t, y)
        f_prime = df_dt(t, y) + df_dy(t, y) * f_val
        y += h * f_val + (h**2 / 2) * f_prime
        t += h
        t_vals.append(t)
        y_vals.append(y)
    return np.array(t_vals), np.array(y_vals)

# Parameters
t0, y0, h, t_end = 1.0, 0.0, 0.1, 2.0
t_taylor, y_taylor = taylor_method_order2(f, df_dt, df_dy, t0, y0, h, t_end)
y_exact = exact_solution(t_taylor)

# Display comparison
print("Taylor Method Order 2:")
print("t\tTaylor\tExact\tError")
for t, yt, ye in zip(t_taylor, y_taylor, y_exact):
    print(f"{t:.1f}\t{yt:.6f}\t{ye:.6f}\t{abs(yt - ye):.6f}")
