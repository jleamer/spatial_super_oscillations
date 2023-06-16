import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

filename = "f=0.63THz_interp.npz"

data = np.load(filename)

Ey = data['Ey'].reshape(10000, 10000)
Ex = data['Ex'].reshape(10000, 10000)
Ez = data['Ez'].reshape(10000, 10000)

incident = Ex[:, 0] * Ex[:, 0].conj() + Ey[:, 0] * Ey[:, 0].conj() + Ez[:, 0] * Ez[:, 0].conj()
y = data['Y']

transmitted = Ex[:, -1] * Ex[:, -1].conj() + Ey[:, -1] * Ey[:, -1].conj() + Ez[:, -1] * Ez[:, -1].conj()

input_power = simpson(incident, y*1e-3)
print(input_power)

output_power = simpson(transmitted, y*1e-3)
print(output_power)

transmittance = output_power / input_power
print("Transmittance: ", transmittance)
comsol = 6.4741e-7 / 216.49
print("Comsol: ", comsol)
print("Relative error: ", np.abs(comsol - transmittance) / comsol)

plt.plot(y, incident)
plt.show()