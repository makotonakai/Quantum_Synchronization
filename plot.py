from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import numpy as np
import config

gamma = config.gamma
num_step = config.num_step
delta_t = config.delta_t

den_mat_list = []

for step in range(num_step):
  upper = config.upper_half(gamma, step, delta_t)
  lower = config.lower_half(gamma, step, delta_t)
  den_mat = np.tensordot(upper,lower,axes=0)
  den_mat_list.append(den_mat)
  print("step {} done".format(step))
  
  
zero_02_re_list = [] # Real part of rho_{0,-1} 
zero_12_re_list = [] # Real part of rho_{1,-1} 
zero_10_re_list = [] # Real part of rho_{1,0} 

zero_02_im_list = [] # Imaginary part of rho_{0,-1} 
zero_12_im_list = [] # Imaginary part of rho_{1,-1} 
zero_10_im_list = [] # Imaginary part of rho_{1,0} 


for den_mat in den_mat_list:

  zero_02_re_list.append(den_mat[1][0][0][0].real)
  zero_12_re_list.append(den_mat[0][1][1][0].real)
  zero_10_re_list.append(den_mat[0][1][0][0].real)
  zero_02_im_list.append(den_mat[1][0][0][0].imag)
  zero_12_im_list.append(den_mat[0][1][1][0].imag)
  zero_10_im_list.append(den_mat[0][1][0][0].imag)
  
fig, ax = plt.subplots(1, 3, figsize=(14,4))

t = np.linspace(0, num_step, num_step)

# Plot
ax[0].plot(t, zero_02_re_list, label='Real part of rho_{0,-1}')
ax[0].plot(t, zero_02_im_list, label='Imaginary part of rho_{0,-1}')

ax[1].plot(t, zero_12_re_list, label='Real part of rho_{1,-1}')
ax[1].plot(t, zero_12_im_list, label='Imaginary part of rho_{1,-1}')

ax[2].plot(t, zero_10_re_list, label='Real part of rho_{1,0}')
ax[2].plot(t, zero_10_im_list, label='Imaginary part of rho_{1,0}')

# Put the legends on each chart
ax[0].legend(loc='best', fontsize=10)
ax[1].legend(loc='best', fontsize=10)
ax[2].legend(loc='best', fontsize=10)

# Put a label on the x axis
ax[0].set_xlabel('# of steps', fontsize=14)
ax[1].set_xlabel('# of steps', fontsize=14)
ax[2].set_xlabel('# of steps', fontsize=14)

# Put a label on the y axis
ax[0].set_ylabel('Population', fontsize=14)
ax[1].set_ylabel('Population', fontsize=14)
ax[2].set_ylabel('Population', fontsize=14)

# Modify the values with ticks
ax[0].tick_params(which='both', direction='in', labelsize=14)
ax[1].tick_params(which='both', direction='in', labelsize=14)
ax[2].tick_params(which='both', direction='in', labelsize=14)

# Put a title to each chart
ax[0].set_title('rho_{0,-1}', fontsize=14)
ax[1].set_title('rho_{1,-1}', fontsize=14)
ax[2].set_title('rho_{1,0}', fontsize=14)

plt.tight_layout()
plt.savefig('./figures/figure_{}steps.png'.format(num_step)) # -----(2)
print("The figure is located in ./figures directory")
# plt.show()