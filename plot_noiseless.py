from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
import numpy as np

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

gamma = 4           # the degree of dissipation
num_step = 15     # The number of steps for the trotterization
dt = 0.05      # the time lapse for each step

delta = 0                          # Δ in the Figure 1
j_0_plus = 2*np.exp(-np.pi*1J/6) # j_{0,+1} in the Figure 1
j_0_neg = -1*np.exp(2*np.pi*1J/6)   # j_{0,-1} in the Figure 1
  
  
def U_pos_0(qc, control, target, t):
    """
    A function to simulate dissipation (energy release from the system to the environment)
  
    Parameters
      qc[QuantumCircuit]: a quantum circuit in Qiskit
    sys[QuantumRegister]: a qubit in the quantum system of interest
    env[QuantumRegister]: an ancillary qubit in the environment(outside of the quantum system of interest)
    
    """
    qc.x(control)
    qc.cu3(-2*np.abs(j_0_plus)*t, np.angle(j_0_plus)-3*np.pi/2, -np.angle(j_0_plus)-np.pi/2, control, target) 
    qc.x(control)
    

def U_neg_0(qc, control, target, t):
    """
    A function to simulate dissipation (energy release from the system to the environment)
  
    Parameters
      qc[QuantumCircuit]: a quantum circuit in Qiskit
    sys[QuantumRegister]: a qubit in the quantum system of interest
    env[QuantumRegister]: an ancillary qubit in the environment(outside of the quantum system of interest)
    
    """
    qc.x(control)
    qc.cu3(-2*np.abs(j_0_neg)*t, np.angle(j_0_neg)-3*np.pi/2, -np.angle(j_0_neg)-np.pi/2, control, target) 
    qc.x(control)
    
    
# Coupling with the environment outside the system
def couple_with_env(qc, sys, env):

  """
  A function to simulate dissipation (energy release from the system to the environment)
  
  Parameters
      qc[QuantumCircuit]: a quantum circuit in Qiskit
    sys[QuantumRegister]: a qubit in the quantum system of interest
    env[QuantumRegister]: an ancillary qubit in the environment(outside of the quantum system of interest)
  """
  qc.cu3(2*np.sqrt(np.arcsin(gamma*dt)), 0, 0, sys, env)
  qc.cx(env, sys)
  
  
# Perform state tomography
def state_tomography(qc, qubit_list):

    """
    A function for the state tomography for the given circuit 
    which the qubit_list are the qubits for the main system
    
    Parameters
        qc[QuantumCircuit]: a quantum circuit
          qubit_list[list]: a list of qubits in the system
          
    """

    qc_tomo = state_tomography_circuits(qc, qubit_list)
    job = execute(qc_tomo, Aer.get_backend('qasm_simulator'), shots=10000)
    result = job.result()
    state_tomo = StateTomographyFitter(result, qc_tomo)
    state_tomo_fit = state_tomo.fit()
    return state_tomo_fit
    
    
# The first run
def full_circuit(num_step):

    """
    Parameters
    gamma_t[float]: degree of dissipation
     num_step[int]: # of steps
    delta_t[float]: time lapse for each step

    Return
    state_tomo_fit[2][2].real[float]: the population of |-1> state
    state_tomo_fit[0][0].real[float]: the population of |0> state
    state_tomo_fit[1][1].real[float]: the population of |1> state

    """
    q = QuantumRegister(2)
    qc = QuantumCircuit(q)

    # Perform Trotterization
    for i in range(num_step):
    
        qa = QuantumRegister(2)
        qc.add_register(qa)

        # White boxes
        qc.rz(delta*dt/2, q[0])
        qc.rz(delta*dt/2, q[1])

        U_pos_0(qc, q[0], q[1], dt/2)
        U_neg_0(qc, q[1], q[0], dt/2)
        U_neg_0(qc, q[1], q[0], dt/2)
        U_pos_0(qc, q[0], q[1], dt/2)
    
        # Green boxes
        couple_with_env(qc, q[0], qa[0])
        couple_with_env(qc, q[1], qa[1])

    state_tomo_fit = state_tomography(qc, [q[0], q[1]])

    return state_tomo_fit


den_mat_list = []

for step in range(num_step):
  den_mat = full_circuit(step)
  den_mat_list.append(den_mat)
  print("step {} done".format(step))
  
  
zero_02_re_list = [] # Real part of rho_{0,-1} 
zero_12_re_list = [] # Real part of rho_{1,-1} 
zero_10_re_list = [] # Real part of rho_{1,0} 

zero_02_im_list = [] # Imaginary part of rho_{0,-1} 
zero_12_im_list = [] # Imaginary part of rho_{1,-1} 
zero_10_im_list = [] # Imaginary part of rho_{1,0} 


for den_mat in den_mat_list:

  zero_02_re_list.append(den_mat[1][0].real)
  zero_12_re_list.append(den_mat[2][1].real)
  zero_10_re_list.append(den_mat[2][0].real)
  zero_02_im_list.append(den_mat[1][0].imag)
  zero_12_im_list.append(den_mat[2][1].imag)
  zero_10_im_list.append(den_mat[2][0].imag)
  
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
plt.savefig('./figures/figure_{}_steps.png'.format(num_step)) # -----(2)
print("The figure is located in ./figures directory")
# plt.show()