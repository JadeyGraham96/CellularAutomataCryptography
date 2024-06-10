# CellularAutomataCryptography

A Python-based implementation of cryptographic techniques using Cellular Automata to enhance data security through complex and unpredictable encryption patterns.

## Introduction

This repository provides an innovative approach to cryptography by leveraging Cellular Automata (CA). Originally conceived by John von Neumann and Stanislaw Ulam in the 1940s, CA are discrete, abstract computational systems capable of generating complex patterns from simple rules. This project explores their application in cryptography to create robust and secure encryption methods.

## Features

- **Cellular Automata-based Encryption**: Utilises the emergent behaviour of CA to generate pseudo-random sequences for encryption and decryption.
- **RSA Key Generation and Digital Signatures**: Implements RSA for secure key management and data integrity verification.
- **Neural Network Integration**: Enhances the complexity of CA rules with a neural network (MLPClassifier) to ensure unpredictability.
- **Visualisation of Cellular Automata Evolution**: Provides functions to visualise the evolution of CA during encryption and decryption.

## Installation

To use this project, clone the repository and install the required Python packages:

```
git clone https://github.com/yourusername/CellularAutomataCryptography.git
cd CellularAutomataCryptography
pip install -r requirements.txt
```
## Usage
### Example Usage
Here's a basic example to demonstrate the encryption and decryption process:

```
import numpy as np
from CellularAutomataCryptography import CellularAutomataCryptography, plot_evolution

# Initialize the cryptographic system
crypto = CellularAutomataCryptography()

# Random plaintext
plaintext = np.random.randint(0, 2, 150)
print("Plaintext:", plaintext)

# Encryption
ciphertext, padding_length, encryption_evolution, iv, mac = crypto.encrypt(plaintext)
print("Ciphertext:", ciphertext)

# Sign the ciphertext
signature = crypto.sign_data(ciphertext.tobytes())
print("Signature:", signature)

# Decryption
recovered_plaintext, decryption_evolution = crypto.decrypt(ciphertext, padding_length, iv, mac)
print("Recovered Plaintext:", recovered_plaintext)

# Verify the signature
is_signature_valid = crypto.verify_signature(ciphertext.tobytes(), signature)
print("Signature valid:", is_signature_valid)

# Verification
is_equal = np.array_equal(plaintext, recovered_plaintext)
print("Plaintext and Recovered Plaintext are equal:", is_equal)

# Plot the cellular automaton evolution during encryption and decryption
plot_evolution(encryption_evolution, 'Encryption Evolution')
plot_evolution(decryption_evolution, 'Decryption Evolution')
```

## Functions
- generate_seed: Generates a secure binary seed using PBKDF2HMAC.
- build_model: Builds and trains a neural network model (MLPClassifier).
- rule_nn: Predicts the next state of the CA using the neural network.
- generate_pseudo_random_sequence: Generates a pseudo-random sequence via CA evolution.
- encrypt: Encrypts plaintext using CA-generated pseudo-random sequences.
- decrypt: Decrypts ciphertext using CA-generated pseudo-random sequences.
- generate_rsa_keys: Generates RSA key pairs.
- sign_data: Signs data using the RSA private key.
- verify_signature: Verifies the signature using the RSA public key.
- plot_evolution: Visualises the evolution of CA during encryption and decryption.

## Requirements
- Python 3.x
- numpy
- matplotlib
- scikit-learn
- cryptography

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
This project is inspired by the pioneering work of John von Neumann, Stanislaw Ulam, and Stephen Wolfram in the field of Cellular Automata, and integrates concepts from modern cryptographic practices.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request with any enhancements or bug fixes.
