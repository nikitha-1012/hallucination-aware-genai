# Cryptography and Access Control

## Symmetric and Asymmetric Encryption

Encryption is the process of converting plaintext data into ciphertext to prevent unauthorized access and maintain confidentiality.

Symmetric Encryption:
- Uses a single shared secret key for both encryption and decryption.
- Faster and computationally efficient for large volumes of data.
- Key distribution is a challenge because both parties must securely exchange the key beforehand.
- Common standard: Advanced Encryption Standard (AES) with key lengths of 128, 192, or 256 bits.

Asymmetric Encryption (Public-Key Cryptography):
- Uses a mathematically linked key pair: a public key for encryption and a private key for decryption.
- Solves the key exchange problem because the public key can be openly shared.
- Slower than symmetric encryption, often used to establish session keys in TLS handshakes and for digital signatures.
- Common algorithms: RSA (Rivest-Shamir-Adleman) and Elliptic Curve Cryptography (ECC).

## Cryptographic Hash Functions and Password Storage

A cryptographic hash function is a one-way deterministic algorithm that maps input data of arbitrary length to a fixed-size digest.

Properties of Secure Hashes:
- Deterministic: Identical input always yields the identical hash.
- Preimage Resistance: Computationally infeasible to reverse the hash to discover the original input.
- Collision Resistance: Computationally infeasible to find two distinct inputs that generate the same hash output.
- Avalanche Effect: A minor change in input produces a drastically different hash output.
- Common secure algorithms: SHA-256 and SHA-3. MD5 and SHA-1 are cryptographically broken and must not be used for security.

Password Storage:
- Passwords must never be stored in plaintext or with simple fast hashes like SHA-256 (which are vulnerable to GPU brute-forcing and rainbow tables).
- Passwords must be salted with a cryptographically random unique value per user to neutralize rainbow tables.
- Passwords must be processed using slow, memory-hard adaptive hashing algorithms such as bcrypt, Argon2, or PBKDF2 with high iteration counts.

## Authentication and Access Control

Authentication verifies the identity of an entity, whereas authorization (access control) determines what resources an authenticated entity is permitted to access.

Multi-Factor Authentication (MFA):
Requires two or more independent authentication factors:
1. Knowledge factor: Something you know (such as a password or PIN).
2. Possession factor: Something you have (such as a hardware security key, authenticator app TOTP, or smart card).
3. Inherence factor: Something you are (such as fingerprint, facial recognition, or biometric scan).

Access Control Principles:
- Principle of Least Privilege: Users, processes, and programs are granted only the minimum access rights and permissions required to perform their authorized functions.
- Role-Based Access Control (RBAC): Access permissions are assigned to specific job roles rather than individual user accounts. Users are assigned to roles based on their organizational responsibilities.
- Defense-in-Depth: Layering multiple independent security controls so that if one defensive layer fails, subsequent layers prevent compromise.
