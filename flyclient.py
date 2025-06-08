from zcash_client import ZcashClient
from zcash_mmr import Tree, Node
from ancestry_proof import NodeType, AncestryNode, AncestryProof
import requests
import math
import numpy as np
from secrets import SystemRandom
import json
import equihash

CONF_PATH = "zcash.conf"
HEARTWOOD_HEIGHT = 903000

DELTA = 2**(-10)
LAMBDA = 50
C = 0.5
L = 50

SOL_K = 9
SOL_N = 200

_secure_random = SystemRandom()

class FlyclientProof:
    client: ZcashClient
    enable_logging: bool
    blockchaininfo: dict
    tip_height: int
    branch_id: str
    activation_height: int
    upgrade_name: str
    peaks: list[int]
    blocks_to_sample: list[int]
    peak_indices: dict[int, int]
    sample_peaks: dict[int, list[int]]
    ancestry_paths: dict[int, list]
    inclusion_paths: dict[int, list]
    rightmost_leaves: dict[int, int]

    def __init__(self, client: ZcashClient, enable_logging = True):
        self.client = client
        self.enable_logging = enable_logging

        # Get the tip and activation height
        self.blockchaininfo: dict = client.send_command("getblockchaininfo")["result"]
        self.branch_id: str = self.blockchaininfo["consensus"]["chaintip"]
        self.tip_height = int(self.blockchaininfo["blocks"]) - 100
        if self.branch_id in self.blockchaininfo["upgrades"]:
            v = self.blockchaininfo["upgrades"][self.branch_id]
            self.activation_height = v["activationheight"]
            self.upgrade_name = str(v["name"]).lower()
        
        if self.activation_height == 0:
            if enable_logging: print("Activation height not found.")
            return False
        elif enable_logging:
            print(f"Upgrade name: {self.upgrade_name}")
            print(f"Activation height: {self.activation_height}")

        mmr = Tree([], self.activation_height)

        # Get the peaks at the tip
        self.peaks = mmr.peaks_at(self.tip_height - 1)
        peak_heights = mmr.peak_heights_at(self.tip_height - 1)
        if enable_logging:
            print(f"Peaks: {self.peaks}")
            print(f"Peak heights: {peak_heights}")
            print(f"Node count: {mmr.node_count_at(self.tip_height - 1)}")

        # Choose random blocks 
        # TODO: Parametrize
        self.blocks_to_sample = blocks_to_sample(self.activation_height, self.tip_height, 10, 10)
        if enable_logging: print(f"Block headers to sample: {self.blocks_to_sample}")

        self.sample_peaks = dict()
        self.ancestry_paths = dict()
        self.peak_indices = dict()
        self.inclusion_paths = dict()
        self.rightmost_leaves = dict()
        for block_height in self.blocks_to_sample:
            # Compute the ancestry proof indices (aka inclusion proof of rightmost leaf in tip MMR)
            # Each block is the rightmost leaf of the MMR rooted in its successor.
            rightmost_leaf_node_index = mmr.node_index_of_block(block_height - 1)
            if enable_logging: print(f"Rightmost leaf index for block {block_height}: {rightmost_leaf_node_index}")

            # Inclusion proof of the leaf in the sampled block header
            sample_peaks = mmr.peaks_at(block_height - 1)
            sample_heights = mmr.peak_heights_at(block_height - 1)
            inclusion_path = self.path_to_root(sample_peaks[-1], sample_heights[-1], rightmost_leaf_node_index)
            if enable_logging: print(f"Inclusion path for {block_height}: {inclusion_path}")
            
            # Calculate the peak index and height of the leaf
            peak_index: int = 0
            peak_h: int = 0
            for (j, peak) in enumerate(self.peaks):
                peak_index = j
                if rightmost_leaf_node_index < peak:
                    peak_h = peak_heights[j]
                    break
            
            if enable_logging: 
                print(f"Peak index for block {block_height}: {peak_index}")
                print(f"Node count at block {block_height}: {mmr.node_count_at(block_height - 1)}")

            # Calculate the path from the leaf to the peak
            ancestry_path = self.path_to_root(self.peaks[peak_index], peak_h, rightmost_leaf_node_index)
            if enable_logging: print(f"Siblings for block header {block_height}: {ancestry_path}")

            self.sample_peaks[block_height] = sample_peaks
            self.inclusion_paths[block_height] = inclusion_path
            self.peak_indices[block_height] = peak_index
            self.rightmost_leaves[block_height] = rightmost_leaf_node_index
            self.ancestry_paths[block_height] = ancestry_path

    @staticmethod
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
    
    def calculate_total_download_size_bytes(self) -> int:
        # version, hashPrevBlock, merkleRoot, blockCommitments, nTime, nBits, nonce, solutionSize, solution
        header_size = 4 + 32 + 32 + 32 + 4 + 4 + 32 + 3 + 1344      
        node_size = 244
        # Tip height, activation height, consensus branch id, upgrade name
        blockchaininfo_size = 4 + 4 + 8 + len(self.upgrade_name)
        authdataroot_size = 32

        nodes_to_download = self.peaks.copy()
        for _, l in self.sample_peaks.items():
            nodes_to_download += l
        for _, l in self.inclusion_paths.items():
                nodes_to_download += l
        for _, l in self.ancestry_paths.items():
                nodes_to_download += l
        for _, v in self.rightmost_leaves.items():
            nodes_to_download.append(v)
        
        nodes_to_download = set(nodes_to_download)

        return (
            (header_size + authdataroot_size) * (len(self.blocks_to_sample) + 1)
            + node_size * len(nodes_to_download) 
            + blockchaininfo_size
        )
    
    def download_header(self, height) -> dict | None:
        try:
            header = self.client.download_header(height, True)
        except requests.exceptions.RequestException as e:
            print(f"Response error when downloading chain tip: {e.response}")
            return None
        return header

    def download_peaks(self, peak_indices: list) -> list[Node] | None:
        peak_nodes: list[Node] = []
        for i in peak_indices:
            try:
                node_json = client.download_node(self.upgrade_name, i, True)
            except requests.exceptions.RequestException as e:
                print(f"Response error when downloading peak node {i}: {e.response}")
                return None
            peak_nodes.append(Node.from_dict(node_json))
        return peak_nodes

    def download_ancestry_proof(self, nodes: list) -> list[AncestryNode] | None:
        siblings: list[AncestryNode] = []
        ancestry_node: AncestryNode = None
        for n, t in nodes:
            try:
                json_obj = client.download_node(self.upgrade_name, n, True)
            except requests.exceptions.RequestException as e:
                print(f"Response error when downloading history node {n}: {e.response}")
                return None
            ancestry_node = AncestryNode(n, Node.from_dict(json_obj), t)
            siblings.append(ancestry_node)
        return siblings
    
    def download_and_verify(self) -> bool:
        tip_header = self.download_header(self.tip_height)
        if tip_header is None: return False

        if self.enable_logging: 
            print("Tip header:")
            print(json.dumps(tip_header, indent=1))
        
        if verify_pow(tip_header) is False:
            print("Fatal: chaintip PoW verification failed")
            return False
        
        # Download peaks at chain tip
        peak_nodes = self.download_peaks(self.peaks)
        if peak_nodes is None: return False
        
        for block_height in self.blocks_to_sample:
            # Download header
            header = self.download_header(block_height)
            if header is None: return False

            if self.enable_logging: 
                print(f"Sampled header at height {block_height}:")
                print(json.dumps(tip_header, indent=1))

            # Verify block PoW
            if verify_pow(header) is False:
                print(f"Fatal: PoW verification failed for block {block_height}")
                return False

            # TODO: authdataroot
            
            # Download leaf node
            try:
                leaf_node = Node.from_dict(client.download_node(self.upgrade_name, self.rightmost_leaves[block_height], True))
            except requests.exceptions.RequestException as e:
                print(f"Response error when downloading rightmost leaf node: {e.response}")
                return False
            
            # Download peaks at sampled block
            peak_nodes_at_sample = self.download_peaks(self.sample_peaks[block_height])
            if peak_nodes_at_sample is None: return False

            # Download MMR nodes for inclusion proof in sampled header
            siblings_at_sample = self.download_ancestry_proof(self.inclusion_paths[block_height])
            if siblings_at_sample is None: return False

            # Download MMR nodes for ancestry proof
            siblings = self.download_ancestry_proof(self.ancestry_paths[block_height])
            if siblings is None: return False

            # Check chain history root for sampled header
            inclusion_proof = AncestryProof(leaf_node, peak_nodes_at_sample, siblings_at_sample, len(peak_nodes_at_sample) - 1)
            if not inclusion_proof.verify_chain_history_root(header["chainhistoryroot"], self.branch_id):
                print(f"Fatal: sample chainhistoryroot verification failed for block at height {block_height}")
                return False

            # Recompute chain history root from node chain and peaks
            ancestry_proof = AncestryProof(leaf_node, peak_nodes, siblings, self.peak_indices[block_height])
            if not ancestry_proof.verify_chain_history_root(tip_header["chainhistoryroot"], self.branch_id):
                print(f"Fatal: tip chainhistoryroot verification failed for block at height {block_height}")
                return False
        return True

# PoW
def verify_pow(block_header: dict) -> bool:
    pow_sol = bytes.fromhex(block_header["solution"])
    pow_header = (
        int(block_header["version"]).to_bytes(4, 'little', signed=True) 
        + bytes.fromhex(block_header["previousblockhash"] )
        + bytes.fromhex(block_header["merkleroot"])
        + bytes.fromhex(block_header["blockcommitments"])
        + int(block_header["time"]).to_bytes(4, 'little', signed=False)
        + bytes.fromhex(block_header["bits"])
        + bytes.fromhex(block_header["nonce"])
    )

    return equihash.verify(SOL_N, SOL_K, pow_header, pow_sol)

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
    
    proof = FlyclientProof(client, True)
    print(f"Total proof download size: {proof.calculate_total_download_size_bytes()}")
    if proof.download_and_verify() is True:
        print("Success!")