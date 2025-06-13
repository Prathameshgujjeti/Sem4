import Crypto
import binascii
import datetime
import collections
from Crypto.PublicKey import RSA
from Crypto import Random
from Crypto.Hash import SHA
from Crypto.Signature import PKCS1_v1_5

class Client:
    def __init__(self):
        random = Crypto.Random.new().read
        self._private_key = RSA.generate(1024, random)
        self._public_key = self._private_key.publickey()
        self._signer = PKCS1_v1_5.new(self._private_key)

    @property
    def identity(self):
        return binascii.hexlify(self._public_key.exportKey(format='DER')).decode('ascii')

class Transaction:
    def __init__(self, sender, receiver, value):
        self.sender = sender
        self.receiver = receiver
        self.value = value
        self.time = datetime.datetime.now()

    def to_dict(self):
        if self.sender == "Genesis":
            identity = "Genesis"
        else:
            identity = self.sender.identity
        return collections.OrderedDict({
            'sender': identity,
            'receiver': self.receiver,
            'value': self.value,
            'time': self.time
        })

    def sign_transaction(self):
        if self.sender == "Genesis":
            return None
        private_key = self.sender._private_key
        signer = PKCS1_v1_5.new(private_key)
        h = SHA.new(str(self.to_dict()).encode('utf8'))
        return binascii.hexlify(signer.sign(h)).decode('ascii')

class Block:
    def __init__(self):
        self.verified_transactions = []
        self.previous_block_hash = ""
        self.Nonce = ""

    @staticmethod
    def display_transaction(transaction):
        d = transaction.to_dict()
        print("Sender: " + d['sender'])
        print('-----')
        print("Receiver: " + d['receiver'])
        print('-----')
        print("Value: " + str(d['value']))
        print('-----')
        print("Time: " + str(d['time']))
        print('-----')

def dump_blockchain(blockchain):
    print("Number of blocks in chain: " + str(len(blockchain)))
    for x in range(len(blockchain)):
        block_temp = blockchain[x]
        print("block #" + str(x))
        for transaction in block_temp.verified_transactions:
            Block.display_transaction(transaction)
            print('-' * 20)
        print("=" * 30)

# Create a client
Ninad = Client()

# Create the genesis transaction
t0 = Transaction(
    "Genesis",
    Ninad.identity,
    500.0
)

# Create the genesis block
block0 = Block()
block0.previous_block_hash = None
block0.Nonce = None
block0.verified_transactions.append(t0)

# Create the blockchain and add the genesis block
TPCoins = []
TPCoins.append(block0)

# Display the blockchain
dump_blockchain(TPCoins)