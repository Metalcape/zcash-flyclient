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

    def __init__(self, last_leaf: Node, peaks: list[Node], path_to_peak: list[AncestryNode], peak_index: int):
        self.rightmost_leaf = last_leaf
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

def validate_min_size_proof(mmr: Tree, nodes: dict[int, Node], expected_hash: str, branch_id: str) -> bool: 
    last_key = max(nodes.keys())
    end_height = nodes[last_key].nLatestHeight
    peaks_with_heights = [(i, h) for i, h in zip(mmr.peaks_at(end_height), mmr.peak_heights_at(end_height))]
    
    peak_nodes = {
        index: { 
            "node": derive_peak_at(index, height, nodes, mmr),
            "height": height
        } for index, height in peaks_with_heights
    }
    
    proof = AncestryProof(
        peak_nodes[max(peak_nodes.keys())],
        [item["node"] for _, item in peak_nodes.items()],
        None,
        0
    )
    
    return proof.verify_root_from_peaks(expected_hash, branch_id)

def derive_peak_at(peak_index: int, peak_height: int, nodes: dict[int, Node], tree: Tree) -> Node | None:
    assert peak_height >= 0
    if peak_index in nodes.keys():
        return nodes[peak_index]
    else:
        return make_parent(
            derive_peak_at(tree.left_child(peak_index, peak_height), peak_height-1, nodes, tree), 
            derive_peak_at(tree.right_child(peak_index), peak_height-1, nodes, tree)
        )
