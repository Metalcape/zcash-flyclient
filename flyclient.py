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

# Equihash parameters
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
    peak_heights: list[int]
    blocks_to_sample: list[int]
    upgrade_names_of_samples: dict[int, str]
    peak_indices: dict[int, int]
    sample_peaks: dict[int, list[int]]
    sample_peak_heights: dict[int, list[int]]
    inclusion_paths: dict[int, list]
    ancestry_paths: dict[int, list]
    extended_ancestry_paths: dict[int, dict]
    rightmost_leaves: dict[int, int]

    def __init__(self, client: ZcashClient, override_chain_tip: int | None = None, enable_logging = True, c: float = 0.5, L: int = 100):
        self.client = client
        self.enable_logging = enable_logging

        # Check that c and L are valid
        if c <= 0 or c >= 1:
            print("c should be between 0 and 1. Falling back to c = 0.5")
            c = 0.5
        if L <= 0:
            print("L should be positive. Falling back to L = 100")
            L = 100

        # Get the tip and activation height
        self.blockchaininfo: dict = client.send_command("getblockchaininfo")["result"]
        if override_chain_tip is not None and override_chain_tip > HEARTWOOD_HEIGHT:
            self.tip_height = int(override_chain_tip)
        else:
            self.tip_height = int(self.blockchaininfo["blocks"]) - 100
        
        (self.branch_id, self.upgrade_name, self.activation_height) = self.get_network_upgrade_of_block(self.tip_height)
        
        if self.activation_height == 0:
            if enable_logging: print("Activation height not found.")
            return False
        elif enable_logging:
            print(f"Upgrade name: {self.upgrade_name}")
            print(f"Activation height: {self.activation_height}")

        mmr = Tree([], self.activation_height)

        # Get the peaks at the tip
        self.peaks = mmr.peaks_at(self.tip_height - 1)
        self.peak_heights = mmr.peak_heights_at(self.tip_height - 1)
        if enable_logging:
            print(f"Peaks: {self.peaks}")
            print(f"Peak heights: {self.peak_heights}")
            print(f"Node count: {mmr.node_count_at(self.tip_height - 1)}")

        # Choose random blocks 
        self.blocks_to_sample = blocks_to_sample(HEARTWOOD_HEIGHT + 1, self.tip_height, c, L)
        self.blocks_to_sample.sort()
        if enable_logging: print(f"Block headers to sample: {self.blocks_to_sample}")

        self.sample_peaks = dict()
        self.sample_peak_heights = dict()
        self.ancestry_paths = dict()
        self.peak_indices = dict()
        self.inclusion_paths = dict()
        self.rightmost_leaves = dict()
        self.extended_ancestry_paths = dict()
        self.upgrade_names_of_samples = dict()
        for block_height in self.blocks_to_sample:
            (_, upgrade, activation_height) = self.get_network_upgrade_of_block(block_height)

            # Edge case: if the activation block is sampled, we must take the rightmost leaf
            # of the previous upgrade's MMR tree
            if activation_height == block_height:
                (_, upgrade, activation_height) = self.get_network_upgrade_of_block(block_height - 1)
                
            mmr = Tree([], activation_height)
            if enable_logging: print(f"Processing block {block_height} ({upgrade})")

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
            if upgrade == self.upgrade_name:
                last_block = self.tip_height - 1
            else:
                next_activation_height = 0
                (_, next_activation_height) = self.next_upgrade(upgrade)
                last_block = next_activation_height - 1
            (peak_index, peak_h) = self.get_peak_index_and_height(mmr, last_block, block_height - 1)
            
            if enable_logging: 
                print(f"Peak index for block {block_height}: {peak_index}")
                print(f"Node count at block {block_height}: {mmr.node_count_at(block_height - 1)}")

            # Calculate the path from the leaf to the peak
            ancestry_path = self.path_to_root(sample_peaks[peak_index], peak_h, rightmost_leaf_node_index)
            if enable_logging: print(f"Siblings for block header {block_height}: {ancestry_path}")

            # Check if the ancestry path belongs to the latest upgrade. If not, we must keep
            # climbing the MMR tree(s) until we get to the chaintip root.
            extended_path = dict()
            current_upgrade = upgrade
            current_activation_height = activation_height
            while current_upgrade != self.upgrade_name:
                # Go up one network upgrade
                (current_upgrade, current_activation_height) = self.next_upgrade(current_upgrade)
                mmr = Tree([], current_activation_height)

                # Get the last block covered by the network upgrade
                if current_upgrade == self.upgrade_name:
                    last_block = self.tip_height - 1
                else:
                    (_, next_activation_height) = self.next_upgrade(current_upgrade)
                    last_block = next_activation_height - 1

                # Get peak index and height of the first leaf
                (ext_peak_index, ext_peak_h) = self.get_peak_index_and_height(mmr, last_block, current_activation_height)

                # Get the path to the root
                path = self.path_to_root(ext_peak_index, ext_peak_h, 0)
                extended_path[current_upgrade] = path

            self.sample_peaks[block_height] = sample_peaks
            self.sample_peak_heights[block_height] = sample_heights
            self.inclusion_paths[block_height] = inclusion_path
            self.peak_indices[block_height] = peak_index
            self.rightmost_leaves[block_height] = rightmost_leaf_node_index
            self.ancestry_paths[block_height] = ancestry_path
            self.extended_ancestry_paths[block_height] = extended_path
            self.upgrade_names_of_samples[block_height] = upgrade

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

    @staticmethod
    def get_peak_index_and_height(tree: Tree, rightmost_block: int, block: int) -> tuple[int, int]:
        peak_index: int = 0
        peak_h: int = 0
        peaks = tree.peaks_at(rightmost_block)
        peak_heights = tree.peak_heights_at(block)
        leaf = tree.node_index_of_block(block)
        for (j, peak) in enumerate(peaks):
            peak_index = j
            if leaf < peak:
                peak_h = peak_heights[j]
                break
        return (peak_index, peak_h)
    
    def next_upgrade(self, upgrade_name: str) -> tuple[str, int] | None:
        for k, v in self.blockchaininfo['upgrades'].items():
            if str(v['name']).lower() == upgrade_name:
                keys = list(self.blockchaininfo['upgrades'].keys())
                next_key_index = keys.index(k) + 1
                if len(keys) > next_key_index:
                    next_key = keys[next_key_index]
                    return (
                        str(self.blockchaininfo['upgrades'][next_key]['name']).lower(), 
                        self.blockchaininfo['upgrades'][next_key]['activationheight']
                    )
                else:
                    return (None, None)
                
        return (None, None)
    
    def get_network_upgrade_of_block(self, height) -> tuple[str, str, int]:
        branch_id = next(iter(self.blockchaininfo['upgrades']))
        for k, v in self.blockchaininfo['upgrades'].items():
            if v['activationheight'] > height:
                break
            else:
                branch_id = k
        
        return (
            str(branch_id), 
            str(self.blockchaininfo['upgrades'][branch_id]['name']).lower(), 
            self.blockchaininfo['upgrades'][branch_id]['activationheight']
        )
    
    def calculate_total_download_size_bytes(self, cache_nodes = True, derive_parents = True) -> int:
        # version, hashPrevBlock, merkleRoot, blockCommitments, nTime, nBits, nonce, solutionSize, solution
        header_size = 4 + 32 + 32 + 32 + 4 + 4 + 32 + 3 + 1344      
        node_size = 244
        # Tip height, activation height, consensus branch id, upgrade name
        blockchaininfo_size = 4 + 4 + 8 + len(self.upgrade_name)
        authdataroot_size = 32

        upgrades_needed = set(self.upgrade_names_of_samples.values())
        assert self.upgrade_name in upgrades_needed

        nodes: dict[str, list] = dict()
        heights: dict[str, list] = dict()
        nodes_with_heights: dict[str, list[tuple]] = dict()
        for k in upgrades_needed:
            nodes[k] = []
            heights[k] = []

        nodes[self.upgrade_name] += self.peaks
        heights[self.upgrade_name] += self.peak_heights
        for k, l in self.sample_peaks.items():
            nodes[self.upgrade_names_of_samples[k]] += l
            heights[self.upgrade_names_of_samples[k]] += list(range(0, len(l)))
        for k, l in self.inclusion_paths.items():
            nodes[self.upgrade_names_of_samples[k]] += [n[0] for n in l]
            heights[self.upgrade_names_of_samples[k]] += list(range(0, len(l)))
        for k, l in self.ancestry_paths.items():
            nodes[self.upgrade_names_of_samples[k]] += [n[0] for n in l]
            heights[self.upgrade_names_of_samples[k]] += list(range(0, len(l)))
        for k, v in self.rightmost_leaves.items():
            nodes[self.upgrade_name].append(v)
            heights[self.upgrade_name].append(0)

        for k in upgrades_needed:
            nodes_with_heights[k] = list(zip(nodes[k], heights[k]))

        # Try to calculate a parent if both children are present
        while derive_parents:
            finished = True
            for k, s in nodes_with_heights.items():
                for i, h in s:
                    if h == 0:
                        continue
                    left_child = (i - 2**h, h - 1)
                    right_child = (i - 1, h - 1)
                    if i in nodes[k] and left_child in s and right_child in s:
                            finished = False
                            nodes[k] = [j for j in nodes[k] if j != i]
            if finished:
                break   
        
        # Count nodes, cache duplicates if requested
        total_node_count = 0
        for _, s in nodes.items():
            if cache_nodes:
                total_node_count += len(set(s))
            else:
                total_node_count += len(s)

        return (
            (header_size + authdataroot_size) * (len(self.blocks_to_sample) + 1)
            + node_size * total_node_count 
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
def sample(n: int, min: int, max: int, delta: float):
    if min <= 1:
        raise ValueError("min must be greater than 1.")
    
    u_min = np.log(min - 1) / np.log(delta)
    u_max = np.log(max - 1) / np.log(delta)
    u_samples = np.array([_secure_random.uniform(u_min, u_max) for _ in range(n)])
    return 1 + delta**u_samples

def blocks_to_sample(activation_height: int, chaintip: int, c: float, L: int):
    # probability of failure is bounded by 2**(-lambda)
    LAMBDA = 50
    # n = chain length
    N = chaintip
    # C = attacker success probability
    # L = estimate of chain difficulty
    # L = delta*n = c**k * n
    # L default is set to the usual size of the non finalized state after sync
    DELTA = L/N
    K = math.log(DELTA, c)

    m = math.ceil(LAMBDA / math.log(1 - (1 / math.log(DELTA, c)), 0.5))
    p_max = (1 - (1/K)) ** m

    # Security property
    assert p_max <= 2 ** (-LAMBDA)

    deterministic = [i for i in range(chaintip - L, chaintip)]
    random = sample(m, activation_height, chaintip - L, DELTA)
    return np.concatenate((random, np.asarray(deterministic, dtype=np.float64))).round().astype(int).tolist()

if __name__ == "__main__":
    client = ZcashClient.from_conf(CONF_PATH)
    # client = ZcashClient("flyclient", "", 8232, "127.0.0.1")
    
    proof = FlyclientProof(client, True)
    print(f"Total proof download size: {proof.calculate_total_download_size_bytes()}")
    if proof.download_and_verify() is True:
        print("Success!")