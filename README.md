# Questions:

Carefully explain in your own words: 

##### Why can quantum computing provide speedups for simulating quantum systems? 

It is well know that classical and quantum information behave differently -- classical information is a subset of quantum information. Thus, in studying quantum systems, a key advantage of quantum computers is the capacity to prepare and store a quantum state using quantum and not classical storage. As quantum systems can be stored directly on qubits with limited overhead, we require 2^{n + 1} - 2 floats to store the same information. Similarly, to manipulate the information, in the worst case we must apply matrix multiplication to the full state space, which is exponentially hard. In studying quantum systems, it is well know that we can evolve a prepared state using O(n^2*t) operations. So, the storage of the information is polynomial and the evolution is polynomial. This means that, provided the Hamiltonian of a system and the initial state can be expressed with a polynomial number of parameters, we can study quantum systems by digitally mapping their dynamics to the evolution of a quantum state on a quantum system. 

A key challenge is in encoding the exponential complexity of the state using gate operations in polynomial time. This is where VQE comes in, although you still require an efficient ansatz and efficient method of training (avoiding barren plateaus and ensuring the ansatz expresses the ground state correctly).  For studying systems at low temperatures, these techniques are particularly useful as the ground state is a realistic initial state for a system. 



##### Describe the variational quantum eigensolver algorithm. How can it be used to prepare eigenstates of input Hamiltonians? How can we compute expectation value without increasing circuit depth? How does the number of samples needed to compute an expectation value scale with the target error and number of local terms in the Hamiltonian?

The variational quantum eigensolver algorithm (VQE) exploits the variational principle in quantum mechanics: for states with given Hamiltonian, the measured energy of the state will be equal to or larger than the ground state of the Hamiltonian. VQE takes as an input a parameterized circuit ansatz for the ground state -- which much contain the ground state within the parameter state space -- a Hamiltonian for which to find the ground state, and an initial guess of the ansatz parameters. The algorithm iterates through guesses for the parameters in order to minimize the expectation value of the Hamiltonian provided. The quantum computer is used to measure the energy of the input state, and a classical computer is used to run an optimization algorithm which, using the quantum cost function, minimizes the parameters. Once minimized, the input parameters and the cost function provide an efficient method to prepare the groundstate and the compute ground energy of the Hamiltonian, respectively. 

One aspect of VQE is the ansatz circuit -- given exponential parameters, the ground state can always be found. Of course, this defeats the purpose of the algorithm as an efficient cost function avoiding the exponential cost of evaluating the expectation classically. So, an ansatz with minimal parameters must be input, which generally requires some fore knowledge of the expected solution space. 

A second vital component is the cost function evaluation. Generally the Hamiltonian is provided as a sum of Pauli strings -- terms which can be represented as the Kronecker product of n pauli matrices or the identity matrix. Each of these terms are unitary, and so we can compute the expectation value <img src="https://render.githubusercontent.com/render/math?math=<\psi|P |\psi>"> relatively easily. Unfortunately, the Hamiltonian is given by a sum of these terms which is not itself unitary and thus cannot be be implemented as a circuit. Fortunately, the inner product is linear, so the each term in the expectation can be evaluated separately. The standard VQE requires preparing m different circuits for a Hamiltonian with m terms, and using these m circuits to evaluate E, which is the sum of the expectations and the cost function for each iteration step.  

For two terms which commute qubit-wise, such as XI and IX (ie, each pauli term at each qubit commutes with each other Pauli term on the same qubit) can be measure simultaneous essentially for free. To do so, the terms are simply rotated simultaneously into the Z computational basis so the expectations of XI and IX would be given by the expectations `<ZI>` and `<IZ>`, respectively after applying a Hadamard at the end of the circuit.  For many Hamiltonians, this provides an efficient reduction in the number of circuits required.  

The ideal method for simultaneous measurement is to measure in an entangled basis. This allows for simultaneously measuring any two commuting terms - not constrained to element wise commutation. For instance, XX and YY can be measured simultaneously because they commute overall, but not element-wise. Finding ideal groupings of measurements is an NP-Hard problem, but heuristics provide reasonable approximations of the minimal number of measurements needed for a given Hamiltonian.  Unfortunately,  preparing the entangled measurements requires arbitrary superposition creation which is expensive on noisy or linear (anything except all to all) connected hardware.  This introduces significant noise to an otherwise optimally short circuit. 

For the immediate future, convergence is bounded by noisy operations. In theory, the precision of the result is determined by the number of shots used in the expectation value computation.  As the expectations are determined from the probabilities of measuring a given qubit or combination of qubits, VQE has precision *1/shots*.  Thus, each evaluation requires a total shot cost of Hamiltonian terms *1/precision* number of iterations, which can be quite expensive.  However, VQE is appealing for near-term applications because it trades circuit depth (and thus noise) for total runtime. The ansatz is minimial to prepare the ground state, and is the only cost in the basic VQE.  

The Pauli Measurement basis can be evaluated using a few techniques, but the best referenced methods are rotating the Pauli term into the nearest single Pauli Z expectation (ie, `<XX>` -> `<IZ>`) which generally requires entangling gates, or rotating into a diagonal term in the computation basis (`<XX>` -> `<ZZ>`). This second method requires determining which measurements are in the +1 and -1 eigenspaces of the measurement term, which is not computationally hard.  This second method also has the advantage of requiring only single qubit gates at the end of the circuit, which can generally be regarded as having a trivial cost (at most 2 single qubit gates are required per qubit). 

# Coding challenges (Linked):

### [Question 1](Question1.ipynb):
>Create a six-qubit circuit using PennyLane that is capable of preparing any state of the form a|110000> + b|001100> + c|000011> + d|100100>, where a,b,c,d are real parameters.

Reference: 

- [Universal quantum circuits for quantum chemistry](https://arxiv.org/pdf/2106.13839.pdf)
- [https://pennylane.ai/qml/demos/tutorial_givens_rotations.html](https://pennylane.ai/qml/demos/tutorial_givens_rotations.html)


Two solve this problem we will use double and single excitation Givens rotations, which preserve particle number. The initial state prepared is |001100>, as it has the most overlap with the target initial state. 

```
----|G1|--------|G3|--------------
----|G1|---------||---------------
X---|G1|--|G2|--|G3|--------------
X---|G1|--|G2|---●----------------
----------|G2|--------------------
----------|G2|--------------------
```

G1: <img src="https://render.githubusercontent.com/render/math?math=-2sin^{-1}(a)">

G2: <img src="https://render.githubusercontent.com/render/math?math=-2sin^{-1}(\frac{c}{\sqrt(1-a^2)})">

G3: <img src="https://render.githubusercontent.com/render/math?math=-2sin^{-1}(\frac{d}{\sqrt(1-a^2-c^2)})">

> Your circuit should include gates with free parameters that can be adjusted to prepare any such state. Hint: You may find this demo useful.

Free Parameters - a, b, c, d

1) Verify a^2 + b^2 + c^2 + d^2 = 1
2) Prepare the initial state |001100>
3) Perform a double excitation on 2,3,0,1 with angle G1
4) Perform a double excitation on 2,3,3,4 with angle G2
5) Perform a single excitation on 2,0 controlled by 3 with angle G3
6) Build unit test for randomly generated a, b, c and d.
   
### [Question 2](Question2.ipynb):

Referenc
- [QAOA for MAXCUT](https://pennylane.ai/qml/demos/tutorial_qaoa_maxcut.html)
- [qml.qaoa Documentation](https://pennylane.readthedocs.io/en/stable/code/qml_qaoa.html)
> Now consider the complete graph on six nodes. Use the QAOA module in PennyLane to nstruct the cost Hamiltonian of this graph for the MaxCut problem. Train your circuit to minimize the expectation value of this Hamiltonian. Hint: You may find this demo useful

Hamiltonian:
```
H = 1/2(ZZIIII + ZIZIII + ZIIZII + ZIIIZI + ZIIIIZ +
IZZIII + IZIZII + IZIIZI + IZIIIZ +
IIZZII + IIZIZI + IIZIIZ +
IIIZZI + IIIZIZ +
IIIIZZ)
```

1) Use qml.qaoa to prepare the cost and mixer Hamiltonian Layers
2) Generate a cost function circuit and measurement
3) Iterate over increased layer counts until cost function converges
   



