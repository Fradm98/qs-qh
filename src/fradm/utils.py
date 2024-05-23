from qiskit_ibm_runtime import QiskitRuntimeService
import json
import numpy as np
from ncon import ncon

def call_ibm_account():
    try:
        service = QiskitRuntimeService(name="ibm-ikerbasque")
    except:
        print("Use the token from IBM quantum ikerbasque collab and save the account")
        service = QiskitRuntimeService(channel="ibm_quantum", token="0fb1cd48a37e2722ff37089427a3d8616d7b2287a908d94dfb62520a17e05294adc3458b0ddcd2333774eb5497b16f75965bbb0233cf9c3b0df7343ffa1374b9")  # Credentials may be needed
        service.save_account(channel="ibm_quantum", token="0fb1cd48a37e2722ff37089427a3d8616d7b2287a908d94dfb62520a17e05294adc3458b0ddcd2333774eb5497b16f75965bbb0233cf9c3b0df7343ffa1374b9", name="ibm-ikerbasque", overwrite=True)
        print(f"{[key for key, value in service.saved_accounts().items()]} account saved")
    return service

def least_busy_backend():
    service = call_ibm_account()
    print(service.backends(simulator=False))
    lb = service.least_busy(simulator=False)
    print(lb.name)
    backend = service.get_backend(name=lb.name)
    return backend

def save_jobs(filename: str, jobs: dict):
    with open(f'{filename}.json', 'w') as json_file:
        json.dump(jobs, json_file, indent=4)


def conversion_basis(number, new_basis, max_digits):
    assert number < new_basis**max_digits, (f"Number cannot be expressed in basis: {new_basis} with a maximum number of digits: {max_digits}")
    lsd = []
    quotient = 1
    while quotient != 0:
        quotient = number // new_basis
        number = number / new_basis
        reminder = number % 1
        lsd.append(int(reminder*new_basis))
    if len(lsd) < max_digits:
        len_digits = len(lsd)
        for _ in range(max_digits-len_digits):
            lsd.append(int(0))
    lsd.reverse()
    return lsd

def conversion_list_to_pauli_string(list):
    wrong_basis = False
    for num in list:
        if num > 3:
            wrong_basis = True
    assert wrong_basis == False, ("The list is not in the correct basis for Paulis")
    paulis = ['I','X','Y','Z']
    list_paulis = [paulis[i] for i in list]
    for i in range(1, len(list_paulis)):
        list_paulis[0] = list_paulis[0] + list_paulis[i]
    lp = list_paulis[0]
    return lp

def conversion_list_to_pauli_op(list):
    wrong_basis = False
    for num in list:
        if num > 3:
            wrong_basis = True
    assert wrong_basis == False, ("The list is not in the correct basis for Paulis")
    I = np.array([[1,0],[0,1]])
    X = np.array([[0,1],[1,0]])
    Y = np.array([[0,-1j],[1j,0]])
    Z = np.array([[1,0],[0,-1]])

    paulis = [I,X,Y,Z]
    list_paulis = [paulis[i] for i in list]
    return list_paulis

def generalized_pauli_op(list_paulis):
    n_qubits = len(list_paulis)
    contr = np.array([1]).reshape((1,1))
    for i in range(n_qubits):
        contr = ncon([contr, list_paulis[i]], [[-1,-3],[-2,-4]]).reshape((2**(i+1),2**(i+1)))
    return contr

def find_coeff_pauli(number, matrix):
    n_qubits = int(np.log2(len(matrix)))
    list_string = conversion_basis(number,4,n_qubits)
    list_paulis_string = conversion_list_to_pauli_string(list_string)
    list_paulis = conversion_list_to_pauli_op(list_string)
    pauli_op = generalized_pauli_op(list_paulis)
    coeff = 1/len(matrix)*np.trace(pauli_op @ matrix)
    return coeff, list_paulis_string

def pauli_decomposition(matrix, threshold: float=1e-15):
    n_qubits = int(np.log2(len(matrix)))
    coeffs = [find_coeff_pauli(num,matrix) for num in range(4**n_qubits)]
    coeffs_new = [[c[0],c[1]] for c in coeffs]
    lin_comb = []
    for coeff in coeffs_new:
        if np.abs(coeff[0].real) > threshold:
            lin_comb.append(coeff)
    return lin_comb, coeffs_new
