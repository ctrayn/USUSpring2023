from math import log, sqrt
import matplotlib.pyplot as plt
e = 2.71828

tox = 100e-10
Na = 2e17
vt = 0.7
Vsb = 4
Eox = 3.9*8.854e-12
Esi = 1.05e-10
q = 1.602e-19
T = 293 # K
ni = 5.29e19 * (T/300)**2.54 * e**(-6726/T)

phi = 2 * vt * log(Na/ni,e)
gamma = tox / Eox * sqrt(2 * q * Esi * Na)

Vt = vt + gamma*(sqrt(phi + Vsb) - sqrt(phi))

print(f"Phi {phi} Gamma {gamma} Vt {Vt}")

Vt_vals = []
for i in range(100):
    delta_T = T + i
    ni = 5.29e19 * (delta_T/300)**2.54 * e**(-6726/delta_T)
    Vt_vals.append(ni)

plt.plot(Vt_vals)
plt.show()