// Contracts


pragma solidity ^0.5.0;

contract ContractDemo {
    string message = "Hello Shivam";

    function dispMsg() public view returns (string memory) {
        return message;
    }
}


//Inheritance

// SPDX-License-Identifier: MIT
pragma solidity >=0.4.22 <0.6.0;

// Inheritance Example
contract Parent {
    uint256 internal sum;

    function setValue() external {
        uint256 a = 10;
        uint256 b = 20;
        sum = a + b;
    }
}

contract Child is Parent {
    function getValue() external view returns (uint256) {
        return sum;
    }
}

// Abstract Contract Example
pragma solidity ^0.5.17;

contract Calculator {
    function getResult() external view returns (uint256);
}

contract Test is Calculator {
    constructor() public {}

    function getResult() external view returns (uint256) {
        uint256 a = 1;
        uint256 b = 2;
        uint256 result = a + b;
        return result;
    }
}

// Caller Contract for Inheritance Test
contract Caller {
    Child cc = new Child();

    function testInheritance() public returns (uint256) {
        cc.setValue();
        return cc.getValue();
    }

    function showValue() public view returns (uint256) {
        return cc.getValue();
    }
}

//Abstract Contract 

// SPDX-License-Identifier: MIT
pragma solidity ^0.5.17;

// Abstract contract
contract Calculator {
    function getResult() external view returns (uint256);
}

// Concrete contract implementing abstract function
contract Test is Calculator {
    constructor() public {}

    function getResult() external view returns (uint256) {
        uint256 a = 1;
        uint256 b = 2;
        uint256 result = a + b;
        return result;
    }
}


//constructor

// SPDX-License-Identifier: MIT
pragma solidity ^0.5.0;

// Creating a contract
contract constructorExample {
    string str;

    constructor() public {
        str = "Shankar Narayan College";
    }

    function getValue() public view returns (string memory) {
        return str;
    }
}


//interfaces 


pragma solidity ^0.5.0;

// Interface declaration
interface Calculator {
    function getResult() external view returns (uint);
}

// Contract implementing the interface
contract Test is Calculator {
    constructor() public {}

    function getResult() external view returns (uint) {
        uint a = 1;
        uint b = 2;
        uint result = a + b;
        return result;
    }
}

