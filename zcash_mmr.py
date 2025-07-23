import hashlib
import json
import struct

SERIALIZED_SIZE = 244

# Integer serialization
def serialize_uint32(n: int) -> bytes:
    return n.to_bytes(4, 'little')

def serialize_uint64(n: int) -> bytes:
    return n.to_bytes(8, 'little')

def serialize_uint256(n: int) -> bytes:
    return n.to_bytes(32, 'little')

def serialize_compact_uint(n: int) -> bytes:
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 252:
        # single-byte encoding
        return struct.pack("B", n)
    elif n <= 0xFFFF:
        # prefix 0xFD + 2-byte little-endian
        return b'\xfd' + struct.pack("<H", n)
    elif n <= 0xFFFFFFFF:
        # prefix 0xFE + 4-byte little-endian
        return b'\xfe' + struct.pack("<I", n)
    elif n <= 0xFFFFFFFFFFFFFFFF:
        # prefix 0xFF + 8-byte little-endian
        return b'\xff' + struct.pack("<Q", n)
    else:
        raise ValueError("n is too large (must fit in 64 bits)")

# Hash function
def hash(data, branch_id_hex: str) -> bytes:
    branch_id = int(branch_id_hex, base=16)
    person = bytearray('ZcashHistory', encoding='ascii') + branch_id.to_bytes(4, 'little')
    return hashlib.blake2b(data, digest_size=32, person=person).digest()

class Node:
    # commitments
    hashSubtreeCommitment: bytes
    nEarliestTimestamp: int
    nLatestTimestamp: int
    nEarliestTargetBits: int
    nLatestTargetBits: int
    hashEarliestSaplingRoot: bytes # left child's Sapling root
    hashLatestSaplingRoot: bytes # right child's Sapling root
    nSubTreeTotalWork: int  # total difficulty accumulated within each subtree
    nEarliestHeight: int
    nLatestHeight: int
    nSaplingTxCount: int # number of Sapling transactions in block
    # NU5 only.
    hashEarliestOrchardRoot: None | bytes # left child's Orchard root
    hashLatestOrchardRoot: None | bytes # right child's Orchard root
    nOrchardTxCount: None | int # number of Orchard transactions in block

    consensusBranchId: bytes

    @staticmethod
    def from_dict(data: dict):
        self = Node()
        self.hashSubtreeCommitment = bytes.fromhex(data["subtree_commitment"])
        self.nEarliestTimestamp = data["start_time"]
        self.nLatestTimestamp = data["end_time"]
        self.nEarliestTargetBits = data["start_target"]
        self.nLatestTargetBits = data["end_target"]
        self.hashEarliestSaplingRoot = bytes.fromhex(data["start_sapling_root"])
        self.hashLatestSaplingRoot = bytes.fromhex(data["end_sapling_root"])
        self.nSubTreeTotalWork = int(data["subtree_total_work"], base=16)
        self.nEarliestHeight = data["start_height"]
        self.nLatestHeight = data["end_height"]
        self.nSaplingTxCount = data["sapling_tx"]
        self.hashEarliestOrchardRoot = bytes.fromhex(data["start_orchard_root"]) if int(data["start_orchard_root"], base=16) != 0 else None
        self.hashLatestOrchardRoot = bytes.fromhex(data["end_orchard_root"]) if int(data["end_orchard_root"], base=16) != 0 else None
        self.nOrchardTxCount = data["orchard_tx"]
        self.consensusBranchId = bytes.fromhex(data['consensus_branch_id'])
        return self

    def serialize(self) -> bytes:
        '''Serializes a node with padding to `SERIALIZED_SIZE` bytes.'''
        buf = (self.hashSubtreeCommitment
            + serialize_uint32(self.nEarliestTimestamp)
            + serialize_uint32(self.nLatestTimestamp)
            + serialize_uint32(self.nEarliestTargetBits)
            + serialize_uint32(self.nLatestTargetBits)
            + self.hashEarliestSaplingRoot
            + self.hashLatestSaplingRoot
            + serialize_uint256(self.nSubTreeTotalWork)
            + serialize_compact_uint(self.nEarliestHeight)
            + serialize_compact_uint(self.nLatestHeight)
            + serialize_compact_uint(self.nSaplingTxCount))
        if self.hashEarliestOrchardRoot is not None:
            buf += (self.hashEarliestOrchardRoot
                + self.hashLatestOrchardRoot
                + serialize_compact_uint(self.nOrchardTxCount))
        buf += bytes.fromhex("00" * (SERIALIZED_SIZE - len(buf)))
        return buf
    
    def serialize_for_hashing(self) -> bytes:
        '''Serializes a node without padding'''
        buf = (self.hashSubtreeCommitment
            + serialize_uint32(self.nEarliestTimestamp)
            + serialize_uint32(self.nLatestTimestamp)
            + serialize_uint32(self.nEarliestTargetBits)
            + serialize_uint32(self.nLatestTargetBits)
            + self.hashEarliestSaplingRoot
            + self.hashLatestSaplingRoot
            + serialize_uint256(self.nSubTreeTotalWork)
            + serialize_compact_uint(self.nEarliestHeight)
            + serialize_compact_uint(self.nLatestHeight)
            + serialize_compact_uint(self.nSaplingTxCount))
        if self.hashEarliestOrchardRoot is not None:
            buf += (self.hashEarliestOrchardRoot
                + self.hashLatestOrchardRoot
                + serialize_compact_uint(self.nOrchardTxCount))
        return buf
    
    def to_json(self) -> str:
        ''' Serializes a node to a JSON string'''
        obj = {}
        obj["subtree_commitment"] = self.hashSubtreeCommitment.hex()
        obj["start_time"] = self.nEarliestTimestamp
        obj["end_time"] = self.nLatestTimestamp
        obj["start_target"] = self.nEarliestTargetBits
        obj["end_target"] = self.nLatestTargetBits
        obj["start_sapling_root"] = self.hashEarliestSaplingRoot.hex()
        obj["end_sapling_root"] = self.hashLatestSaplingRoot.hex()
        obj["sapling_tx"] = self.nSaplingTxCount
        obj["subtree_total_work"] = self.nSubTreeTotalWork.to_bytes(length=32, byteorder="big", signed=False).hex()
        obj["start_height"] = self.nEarliestHeight
        obj["end_height"] = self.nLatestHeight
        if self.hashEarliestOrchardRoot is not None:
            obj["start_orchard_root"] = self.hashEarliestOrchardRoot.hex()
            obj["end_orchard_root"] = self.hashLatestOrchardRoot.hex()
            obj["orchard_tx"] = self.nOrchardTxCount
        obj["consensus_branch_id"] = self.consensusBranchId.hex()

        return json.dumps(obj, indent=1)

# Combine nodes
def make_parent(left_child: Node, right_child: Node) -> Node:
    assert left_child.consensusBranchId == right_child.consensusBranchId
    parent = Node()
    parent.hashSubtreeCommitment = hash(left_child.serialize_for_hashing() + right_child.serialize_for_hashing(), left_child.consensusBranchId.hex())
    parent.nEarliestTimestamp = left_child.nEarliestTimestamp
    parent.nLatestTimestamp = right_child.nLatestTimestamp
    parent.nEarliestTargetBits = left_child.nEarliestTargetBits
    parent.nLatestTargetBits = right_child.nLatestTargetBits
    parent.hashEarliestSaplingRoot = left_child.hashEarliestSaplingRoot
    parent.hashLatestSaplingRoot = right_child.hashLatestSaplingRoot
    parent.nSubTreeTotalWork = left_child.nSubTreeTotalWork + right_child.nSubTreeTotalWork
    parent.nEarliestHeight = left_child.nEarliestHeight
    parent.nLatestHeight = right_child.nLatestHeight
    parent.nSaplingTxCount = left_child.nSaplingTxCount + right_child.nSaplingTxCount
    parent.hashEarliestOrchardRoot = left_child.hashEarliestOrchardRoot
    parent.hashLatestOrchardRoot = right_child.hashLatestOrchardRoot
    parent.nOrchardTxCount = (left_child.nOrchardTxCount + right_child.nOrchardTxCount
                            if left_child.nOrchardTxCount is not None and right_child.nOrchardTxCount is not None
                            else None)
    parent.consensusBranchId = left_child.consensusBranchId

    return parent

class Tree:
    __nodes__: list[Node]
    __activation_height__: int

    @classmethod
    def __init__(self, nodes: list[Node], activation_height: int):
        self.__nodes__ = nodes
        self.__activation_height__ = activation_height

    # MMR navigation functions
    @staticmethod
    def right_sibling(index: int, h: int) -> int:
        return index + (2**(h+1) - 1)

    @staticmethod
    def left_sibling(index: int, h: int) -> int:
        return index - (2**(h+1) - 1)

    @staticmethod
    def parent_from_right(index: int) -> int: 
        return index + 1
    
    @staticmethod
    def right_child(index: int) -> int: 
        return index - 1

    @staticmethod
    def parent_from_left(index: int, h: int) -> int: 
        return index + 2**(h+1)
    
    @staticmethod
    def left_child(index: int, h: int) -> int: 
        return index - 2**h


    def insertion_index_of_block(self, height: int) -> int | None:
        diff = height - self.__activation_height__
        if diff < 0:
            return None
        else:
            return diff
        
    def peak_heights_at(self, height: int) -> list[int] | None:
        insertion_index = self.insertion_index_of_block(height)
        if insertion_index is None:
            return None
        
        block_count = insertion_index + 1
        heights = []
        for h in reversed(range(0, 31)):
            mask = 1 << h
            if block_count & mask != 0:
                heights.append(h)
        
        return heights

    def peaks_at(self, height: int) -> list[int] | None:
        insertion_index = self.insertion_index_of_block(height)
        if insertion_index is None:
            return None
        
        block_count = insertion_index + 1
        peaks = []
        total_nodes = 0
        for h in reversed(range(0, 31)):
            mask = 1 << h
            if block_count & mask != 0:
                total_nodes += 2**(h + 1) - 1
                peaks.append(total_nodes - 1)

        return peaks

    def node_count_at(self, height: int) -> int | None:
        peaks = self.peaks_at(height)
        return peaks[-1] + 1 if peaks is not None else None
    
    def node_index_of_block(self, height: int) -> int | None:
        insertion_index = self.insertion_index_of_block(height)
        if insertion_index is None:
            return None
        
        block_count = insertion_index + 1
        total_nodes = 0
        height = 0
        for h in reversed(range(0, 31)):
            mask = 1 << h
            if block_count & mask != 0:
                total_nodes += 2**(h + 1) - 1
                height = h
        
        peak = total_nodes - 1
        return peak - height
    
    def append(self, node: Node):
        self.__nodes__.append(node)
    
    def get(self, node_index: int) -> Node:
        return self.__nodes__[node_index]