const SHA256 = require("crypto-js/sha256");

class Block {
    constructor(index, timestamp, data, previousHash = "") {
        this.index = index;
        this.timestamp = timestamp;
        this.data = data;
        this.previousHash = previousHash;
        this.hash = this.calculateHash();
    }

    calculateHash() {
        return SHA256(
            this.index +
            this.previousHash +
            this.timestamp +
            JSON.stringify(this.data)
        ).toString();
    }
}

class Blockchain {
    constructor() {
        this.chain = [this.createGenesisBlock()];
    }

    createGenesisBlock() {
        return new Block(0, "09/06/2024", "GenesisBlock", "0");
    }

    getLatestBlock() {
        return this.chain[this.chain.length - 1];
    }

    addBlock(newBlock) {
        newBlock.previousHash = this.getLatestBlock().hash;
        newBlock.hash = newBlock.calculateHash();
        this.chain.push(newBlock);
    }

    isChainValid() {
        for (let i = 1; i < this.chain.length; i++) {
            const currentBlock = this.chain[i];
            const previousBlock = this.chain[i - 1];

            if (currentBlock.hash !== currentBlock.calculateHash()) {
                console.log(`Block ${currentBlock.index} hash mismatch.`);
                return false;
            }

            if (currentBlock.previousHash !== previousBlock.hash) {
                console.log(`Block ${currentBlock.index} previous hash mismatch.`);
                return false;
            }
        }
        return true;
    }
}

let myCoin = new Blockchain();

myCoin.addBlock(new Block(1, "09/06/2024", { amount: 4 }));
myCoin.addBlock(new Block(2, "09/06/2024", { amount: 8 }));

console.log(JSON.stringify(myCoin, null, 4));