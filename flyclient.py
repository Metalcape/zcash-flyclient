from zcash_client import ZcashClient
from zcash_mmr import Tree, Node
from ancestry_proof import NodeType, AncestryNode, AncestryProof
import requests
import math
import numpy as np
from secrets import SystemRandom
import json

CONF_PATH = "zcash.conf"
HEARTWOOD_HEIGHT = 903000
DELTA = 2**(-10)

_secure_random = SystemRandom()

# Difficulty
def to_target(x: int) -> int:
    mask = 1 << 23
    if x & mask == mask:
        return 0
    else:
        mantissa = x & (mask - 1)
        exponent = math.floor(x / (1 << 24)) - 3
        return mantissa * 256 ** exponent

def calculate_work(nbits: str):
    value = int(nbits, base=16)
    return math.floor(2 ** 256 / (to_target(value) + 1))

# Block sampling
def sample(n: int, min: int, max: int):
    if min <= 1:
        raise ValueError("min must be greater than 1.")
    
    u_min = np.log(min - 1) / np.log(DELTA)
    u_max = np.log(max - 1) / np.log(DELTA)
    u_samples = np.array([_secure_random.uniform(u_min, u_max) for _ in range(n)])
    return 1 + DELTA**u_samples

def blocks_to_sample(activation_height, chaintip, random_blocks, tip_blocks):
    deterministic = [i for i in range(chaintip - tip_blocks, chaintip)]
    random = sample(random_blocks, activation_height, chaintip - tip_blocks)
    return np.concatenate((random, np.asarray(deterministic, dtype=np.float64))).round().astype(int).tolist()

if __name__ == "__main__":
    client = ZcashClient.from_conf(CONF_PATH)
    # client = ZcashClient("flyclient", "", 8232, "127.0.0.1")

    # 1. Start by getting the consensus branch and the tip block header
    blockchaininfo: dict = client.send_command("getblockchaininfo")["result"]
    branch_id: str = blockchaininfo["consensus"]["chaintip"]
    #tip_block = int(blockchaininfo["blocks"]) - 1
    tip_block = 2876543
    #tip_block = 2726449
    print(f"Branch id: {branch_id}")
    print(f"Tip block: {tip_block}")
    try:
        tip_header = client.download_header(tip_block, True)
    except requests.exceptions.RequestException as e:
        print(f"Response error when downloading chain tip: {e.response}")
        exit(-1)
    
    print("Tip header:")
    print(json.dumps(tip_header, indent=1))
    
    # TODO: 2. authdataroot

    # TODO: Verify header PoW

    # Get the activation height
    activation_height: int
    upgrade_name: str
    if branch_id in blockchaininfo["upgrades"]:
        v = blockchaininfo["upgrades"][branch_id]
        activation_height = v["activationheight"]
        upgrade_name = str(v["name"]).lower()
    
    if activation_height == 0:
        print("Activation height not found.")
        exit(-1)
    else:
        print(f"Upgrade name: {upgrade_name}")
        print(f"Activation height: {activation_height}")

    mmr = Tree([], activation_height)
    peaks = mmr.peaks_at(tip_block)
    peak_heights = mmr.peak_heights_at(tip_block)
    print(f"Peaks: {peaks}")
    print(f"Peak heights: {peak_heights}")
    print(f"Node count: {mmr.node_count_at(tip_block)}")

    # 3. Choose random blocks 
    # TODO: Parametrize
    block_heights = blocks_to_sample(activation_height, tip_block, 10, 10)
    print(f"Block headers to sample: {block_heights}")

    for block_height in block_heights:
        # 4. Download header
        try:
            header = client.download_header(block_height, True)
        except requests.exceptions.RequestException as e:
            print(f"Response error when downloading header {block_height}: {e.response}")
            exit(-1)

        # 5. TODO: Verify header PoW

        # 6. TODO: authdataroot

        # 7. Compute the ancestry proof indices (aka inclusion proof of rightmost leaf in tip MMR)
        # Each block is the rightmost leaf of the MMR rooted in its successor.
        rightmost_leaf_node_index = mmr.node_index_of_block(block_height)
        print(f"Rightmost leaf index for block {block_height}: {rightmost_leaf_node_index}")

        # Calculate the path from the leaf to the peak
        peak_index: int = 0
        peak_h: int = 0
        for (j, peak) in enumerate(peaks):
            peak_index = j
            if rightmost_leaf_node_index < peak:
                peak_h = peak_heights[j]
                break
        
        print(f"Peak index for block {block_height}: {peak_index}")
        print(f"Node count at block {block_height}: {mmr.node_count_at(block_height - 1)}")
        print(f"Node index of block {block_height}: {mmr.node_index_of_block(block_height)}")

        # Calculate the reverse path from the peak to the leaf
        node_path = []
        index = peak
        h = peak_h
        while index > rightmost_leaf_node_index:
            right = Tree.right_child(index)
            left = Tree.left_child(index, h)
            if rightmost_leaf_node_index > left:
                index = right
                node_path.append((left, NodeType.LEFT))
            else:
                index = left
                node_path.append((right, NodeType.RIGHT))
            h = h - 1

        node_path.reverse()
        print(f"Siblings for block header {block_height}: {node_path}")
        
        # 8. Download MMR nodes
        siblings: list[AncestryNode] = []
        ancestry_node: AncestryNode = None
        for n, t in node_path:
            try:
                json_obj = client.download_node(upgrade_name, n, True)
            except requests.exceptions.RequestException as e:
                print(f"Response error when downloading history node {n}: {e.response}")
                exit(-1)
            ancestry_node = AncestryNode(n, Node.from_dict(json_obj), t)
            siblings.append(ancestry_node)
        
        print(f"Siblings for ancestry: {[(s.index, s.type) for s in siblings]}")

        # 8.1: Download leaf node and peaks. TODO: Verify that the block header commits to the leaf
        try:
            leaf_node = Node.from_dict(client.download_node(upgrade_name, rightmost_leaf_node_index, True))
        except requests.exceptions.RequestException as e:
                print(f"Response error when downloading rightmost leaf node: {e.response}")
                exit(-1)

        peak_nodes: list[Node] = []
        for i in peaks:
            try:
                node_json = client.download_node(upgrade_name, i, True)
            except requests.exceptions.RequestException as e:
                print(f"Response error when downloading peak node {i}: {e.response}")
                exit(-1)
            peak_nodes.append(Node.from_dict(node_json))

        # 9. Recompute chain history root from node chain and peaks
        ancestry_proof = AncestryProof(leaf_node, peak_nodes, siblings, peak_index)
        if not ancestry_proof.verify_chain_history_root(tip_header["chainhistoryroot"], branch_id):
            print(f"Fatal: chainhistoryroot verification failed for block at height {block_height}")
            exit(-1)
        
        # 10. TODO: Hash (recomputed) chain history root and (downloaded) auth data root, and check it is equal to to (downloaded) block commitment of sampled block header

        # 11. TODO: Reconstruct chain history root of last block header

        # 12. TODO: Hash (recomputed) chain history root and (downloaded) auth data root, and check it is equal to (downloaded) block commitment of last block header
        
    print("Success")