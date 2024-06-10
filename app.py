import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import os
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
from cryptography.hazmat.primitives import serialization
import hmac
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CellularAutomataCryptography:
    def __init__(self, seed_length=100, key=None):
        self.seed_length = seed_length
        self.key = key if key else os.urandom(32)  # Secure key generation with 32 bytes
        self.initial_seed = self.generate_seed()
        self.seed = self.initial_seed.copy()
        self.model = self.build_model()
        self.private_key, self.public_key = self.generate_rsa_keys()

    def generate_seed(self):
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.seed_length,
            salt=os.urandom(16),  # Securely generated salt
            iterations=100000,
            backend=default_backend()
        )
        seed = kdf.derive(self.key)
        return np.frombuffer(seed, dtype=np.uint8) % 2  # Ensure binary seed

    def build_model(self):
        model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000)
        # Training data should be more meaningful in real applications
        X_train = np.random.randint(0, 2, (1000, self.seed_length))
        y_train = np.random.randint(0, 2, 1000)
        model.fit(X_train, y_train)
        return model

    def rule_nn(self, state):
        rule_input = state.reshape(1, -1)
        prediction = self.model.predict(rule_input)
        new_state = np.zeros_like(state)
        for i in range(1, len(state) - 1):
            new_state[i] = int(prediction[0])
        return new_state

    def generate_pseudo_random_sequence(self, length):
        self.seed = self.initial_seed.copy()
        steps = max(length, 100)
        full_sequence = []
        evolution = []
        for _ in range((length // self.seed_length) + 1):
            self.seed = self.rule_nn(self.seed.copy())
            evolution.append(self.seed.copy())
            full_sequence.extend(self.seed)
        return np.array(full_sequence[:length]), evolution
    
    def pad(self, data):
        padding_length = (self.seed_length - len(data) % self.seed_length) % self.seed_length
        padding = np.zeros(padding_length, dtype=int)
        return np.concatenate([data, padding]), padding_length
    
    def unpad(self, data, padding_length):
        if padding_length == 0:
            return data
        return data[:-padding_length]
    
    def encrypt(self, plaintext):
        try:
            plaintext_padded, padding_length = self.pad(plaintext)
            iv = np.random.randint(0, 2, self.seed_length)
            logger.info(f"IV: {iv}")
            self.initial_seed = iv.copy()
            prng_sequence, evolution = self.generate_pseudo_random_sequence(len(plaintext_padded))
            ciphertext = np.bitwise_xor(plaintext_padded, prng_sequence)
            mac = hmac.new(self.key, ciphertext.tobytes(), hashlib.sha256).digest()
            logger.info("Encryption successful")
            return ciphertext, padding_length, evolution, iv, mac
        except Exception as e:
            logger.error("Encryption failed", exc_info=True)
            raise e
    
    def decrypt(self, ciphertext, padding_length, iv, mac):
        try:
            self.initial_seed = iv.copy()
            prng_sequence, evolution = self.generate_pseudo_random_sequence(len(ciphertext))
            plaintext_padded = np.bitwise_xor(ciphertext, prng_sequence)
            if hmac.new(self.key, ciphertext.tobytes(), hashlib.sha256).digest() != mac:
                raise ValueError("MAC verification failed")
            logger.info("Decryption successful")
            return self.unpad(plaintext_padded, padding_length), evolution
        except Exception as e:
            logger.error("Decryption failed", exc_info=True)
            raise e

    def generate_rsa_keys(self):
        try:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            public_key = private_key.public_key()
            logger.info("RSA key pair generated")
            return private_key, public_key
        except Exception as e:
            logger.error("RSA key generation failed", exc_info=True)
            raise e

    def sign_data(self, data):
        try:
            signature = self.private_key.sign(
                data,
                asym_padding.PSS(
                    mgf=asym_padding.MGF1(hashes.SHA256()),
                    salt_length=asym_padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            logger.info("Data signed successfully")
            return signature
        except Exception as e:
            logger.error("Data signing failed", exc_info=True)
            raise e

    def verify_signature(self, data, signature):
        try:
            self.public_key.verify(
                signature,
                data,
                asym_padding.PSS(
                    mgf=asym_padding.MGF1(hashes.SHA256()),
                    salt_length=asym_padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            logger.info("Signature verification successful")
            return True
        except Exception as e:
            logger.error("Signature verification failed", exc_info=True)
            return False

def plot_evolution(evolution, title):
    plt.figure(figsize=(10, 6))
    plt.imshow(evolution, cmap='binary', interpolation='none')
    plt.title(title)
    plt.xlabel('Cell Index')
    plt.ylabel('Time Step')
    plt.show()

# Example Usage
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
