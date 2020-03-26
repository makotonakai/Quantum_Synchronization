from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
import numpy as np
import matplotlib.pyplot as plt

gamma = 4           # the degree of dissipation
num_step = 27     # The number of steps for the trotterization
delta_t = 0.05      # the time lapse for each step

delta = 0                          # Δ in the Figure 1
j_0_plus = 2*np.exp(-np.pi*1J/6) # j_{0,+1} in the Figure 1
j_0_neg = -1*np.exp(2*np.pi*1J/6)   # j_{0,-1} in the Figure 1

# Coupling with the environment outside the system
def couple_with_env(qc, sys, env):

  """
  A function to simulate dissipation (energy release from the system to the environment)
  
  Parameters
      qc[QuantumCircuit]: a quantum circuit in Qiskit
    sys[QuantumRegister]: a qubit in the quantum system of interest
    env[QuantumRegister]: an ancillary qubit in the environment(outside of the quantum system of interest)
  """
  qc.cu3(2*np.sqrt(np.arcsin(gamma*delta_t)), 0, 0, sys, env)
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
def upper_half(gamma, num_step, delta_t):

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
    qc.rz(delta*delta_t/2, q[0])
    
    # Orange boxes
    qc.u3(-2*abs(j_0_neg)*delta_t, np.angle(j_0_neg)-3*np.pi/2, -np.angle(j_0_neg)-np.pi/2, q[0])
    ########### no red box ##########
    
    qc.u3(-2*abs(j_0_neg)*delta_t, np.angle(j_0_neg)-3*np.pi/2, -np.angle(j_0_neg)-np.pi/2, q[0])

    qc.rz(delta*delta_t/2, q[0])

    # Green boxes
    couple_with_env(qc, q[0], qa[0])

  state_tomo_fit = state_tomography(qc, [q[0]])
  
  return state_tomo_fit
  
  
# The second run
def lower_half(gamma, num_step, delta_t):

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
  q = QuantumRegister(1)
  qc = QuantumCircuit(q)

  # Perform Trotterization
  for i in range(num_step):
    qa = QuantumRegister(1)
    qc.add_register(qa)

    # White boxes
    qc.rz(-delta*delta_t/2, q[0])
    
    # Orange boxes
    qc.u3(-2*abs(j_0_plus)*delta_t, np.angle(j_0_plus)-3*np.pi/2, -np.angle(j_0_plus)-np.pi/2, q[0])

    ########### no red box ##########
    
    qc.u3(-2*abs(j_0_plus)*delta_t, np.angle(j_0_plus)-3*np.pi/2, -np.angle(j_0_plus)-np.pi/2, q[0])

    qc.rz(-delta*delta_t/2, q[0])

    # Green boxes
    couple_with_env(qc, q[0], qa[0])

  state_tomo_fit = state_tomography(qc, [q[0]])
  
  return state_tomo_fit