// Function Overloading

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract OverloadingExample {

    function add(uint256 a, uint256 b) public pure returns (uint256) {
        return a + b;
    }

    function add(string memory a, string memory b) public pure returns (string memory) {
        return string(abi.encodePacked(a, b));
    }
}


//Mathematical Function 


pragma solidity ^0.5.0;

contract Test {
    function CallAddMod() public pure returns (uint) {
        return addmod(7, 3, 3);
    }

    function CallMulMod() public pure returns (uint) {
        return mulmod(7, 3, 3);
    }
}


// Cryptographic Functions

pragma solidity ^0.5.0;

contract Test {
    function callKeccak256() public pure returns (bytes32 result) {
        return keccak256(abi.encodePacked("BLOCKCHAIN"));
    }

    function callSha256() public pure returns (bytes32 result) {
        return sha256(abi.encodePacked("BLOCKCHAIN"));
    }

    function callRipemd() public pure returns (bytes20 result) {
        return ripemd160(abi.encodePacked("BLOCKCHAIN"));
    }
}

//