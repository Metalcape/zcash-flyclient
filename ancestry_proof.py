from zcash_mmr import Tree, Node, hash, make_parent
from enum import Enum
from dataclasses import dataclass

class NodeType(Enum):
    LEFT = 0
    RIGHT = 1

@dataclass
class AncestryNode:
    index: int
    node: Node
    type: NodeType

    def __init__(self, index: int, node: Node, type: NodeType):
        self.index = index
        self.node = node
        self.type = type

class AncestryProof:
    rightmost_leaf: Node
    peaks: list[Node]
    peak_index: int
    path: list[AncestryNode]

    def __init__(self, first_leaf: Node, peaks: list[Node], path_to_peak: list[AncestryNode], peak_index: int):
        self.rightmost_leaf = first_leaf
        self.peaks = peaks
        self.path = path_to_peak
        self.peak_index = peak_index

    def get_root(self) -> Node:
        bag = self.peaks[0]
        for p in self.peaks[1:]:
            bag = make_parent(bag, p)
        return bag
    
    def verify_root_from_peaks(self, root_hash: str, branch_id: str):
        # Calculate and compare root hash
        root = self.get_root()
        # print("Root node:")
        # print(root.to_json())
        # print(f"Branch id: {branch_id}")
        calculated_root_hash = hash(root.serialize_for_hashing(), branch_id)[::-1]
        # print(f"Calculated hash: {calculated_root_hash.hex()}")
        # print(f"Expected root hash: {root_hash}")
        if calculated_root_hash != bytes.fromhex(root_hash):
            print("Root hash mismatch")
            return False
        else:
            return True

    def verify_chain_history_root(self, root_hash: str, branch_id: str) -> bool:
        # For each node in path to peak, combine with previous node
        # skip if path is empty
        if len(self.path) > 0:
            current = self.rightmost_leaf
            for sibling in self.path:
                if sibling.type == NodeType.LEFT:
                    current = make_parent(sibling.node, current)
                else:
                    current = make_parent(current, sibling.node)

            # Reverse to match presentation byte order
            calculated_hash = hash(current.serialize_for_hashing(), branch_id)
            expected_hash = hash(self.peaks[self.peak_index].serialize_for_hashing(), branch_id)
            if calculated_hash != expected_hash:
                print ("Peak node hash mismatch")
                return False
        else:
            # If path is empty, peak must be the last one
            assert self.peak_index == len(self.peaks) - 1
        
        return self.verify_root_from_peaks(root_hash, branch_id)
    
def path_to_root(peak: int, peak_height: int, leaf_index: int) -> list:
    node_path = list()
    index = peak
    h = peak_height
    while index > leaf_index:
        right = Tree.right_child(index)
        left = Tree.left_child(index, h)
        if leaf_index > left:
            index = right
            node_path.append((left, NodeType.LEFT))
        else:
            index = left
            node_path.append((right, NodeType.RIGHT))
        h = h - 1
    node_path.reverse()
    return node_path
