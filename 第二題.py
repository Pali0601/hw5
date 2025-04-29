import numpy as np

def f1(t, u1, u2):
    return (1/3)*(9*u1 + 24*u2 + 5*np.cos(t) - np.sin(t))

def f2(t, u1, u2):
    return (1/3)*(-24*u1 - 51*u2 - 9*np.cos(t) + np.sin(t))

def exact_u1(t):
    return (1/3)*np.exp(-3*t) + np.cos(t)

def exact_u2(t):
    return (1/3)*np.exp(-3*t) - np.cos(t)

def rk4_system(t0, u1_0, u2_0, h, t_end):
    t_vals = [t0]
    u1_vals = [u1_0]
    u2_vals = [u2_0]
    t = t0
    u1 = u1_0
    u2 = u2_0
    while t < t_end - 1e-8:  # to handle floating point edge
        k1_1 = h * f1(t, u1, u2)
        k1_2 = h * f2(t, u1, u2)

        k2_1 = h * f1(t + h/2, u1 + k1_1/2, u2 + k1_2/2)
        k2_2 = h * f2(t + h/2, u1 + k1_1/2, u2 + k1_2/2)

        k3_1 = h * f1(t + h/2, u1 + k2_1/2, u2 + k2_2/2)
        k3_2 = h * f2(t + h/2, u1 + k2_1/2, u2 + k2_2/2)

        k4_1 = h * f1(t + h, u1 + k3_1, u2 + k3_2)
        k4_2 = h * f2(t + h, u1 + k3_1, u2 + k3_2)

        u1 += (k1_1 + 2*k2_1 + 2*k3_1 + k4_1) / 6
        u2 += (k1_2 + 2*k2_2 + 2*k3_2 + k4_2) / 6
        t += h

        t_vals.append(t)
        u1_vals.append(u1)
        u2_vals.append(u2)
    
    return np.array(t_vals), np.array(u1_vals), np.array(u2_vals)

# Run with h = 0.05
t_05, u1_05, u2_05 = rk4_system(0, 1/4, 1/2, 0.05, 1)
# Run with h = 0.1
t_10, u1_10, u2_10 = rk4_system(0, 1/4, 1/2, 0.1, 1)

# Compare results
print("h = 0.05:")
print("t\tu1(num)\tu1(exact)\tu2(num)\tu2(exact)")
for t, u1, u2 in zip(t_05, u1_05, u2_05):
    print(f"{t:.2f}\t{u1:.6f}\t{exact_u1(t):.6f}\t{u2:.6f}\t{exact_u2(t):.6f}")

print("\n" + "="*50 + "\n")

print("h = 0.1:")
print("t\tu1(num)\tu1(exact)\tu2(num)\tu2(exact)")
for t, u1, u2 in zip(t_10, u1_10, u2_10):
    print(f"{t:.2f}\t{u1:.6f}\t{exact_u1(t):.6f}\t{u2:.6f}\t{exact_u2(t):.6f}")
