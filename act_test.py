import numpy as np

zeta_a = 0.7
omega_a = 2*np.pi*0.5

u_e = np.zeros((2,1))
A_act = np.array([[0,1],[-(omega_a**2),-2*zeta_a*omega_a]])
B_act = np.array([[0],[omega_a**2]])
u_cmd = 0.2

print(u_e)
print(A_act)
print(B_act)
for ii in range(1000):
    u_e_dot = A_act@u_e + B_act*u_cmd
    u_e += u_e_dot*0.01
    print(u_e_dot)
    print(u_e)