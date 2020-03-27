from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from plot_noiseless import *
import numpy as np

num_step = 4

for step in range(num_step):

    print("step {}".format(step))
    
    matrix = full_circuit(step)
    print(matrix.shape)