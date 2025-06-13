import hashlib

def sha256(message):
    return hashlib.sha256(message.encode("ascii")).hexdigest()

def mine(message, difficulty=1):
    prefix = "1" * difficulty
    for i in range(1000):
        digest = hashlib.sha256(f"{message}{i}".encode()).hexdigest()
        if digest.startswith(prefix):
            print(f"After {i} iterations found nonce: {digest}")
            break

mine("test message", 2)