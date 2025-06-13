import datetime
import hashlib

# Create a class with two functions
class Block:
    def __init__(self, data, previous_hash):
        self.timestamp = datetime.datetime.now(datetime.timezone.utc)
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.calc_hash()

    def calc_hash(self):
        sha = hashlib.sha256()
        hash_str = self.data.encode("utf-8")
        sha.update(hash_str)
        return sha.hexdigest()

# Instantiate the class
blockchain = [Block("First block", "0")]
blockchain.append(Block("Second block", blockchain[0].hash))
blockchain.append(Block("Third block", blockchain[1].hash))

# Dumping the blockchain
for block in blockchain:
    print(
        f"Timestamp: {block.timestamp}\n"
        f"Data: {block.data}\n"
        f"Previous Hash: {block.previous_hash}\n"
        f"Hash: {block.hash}\n"
    )