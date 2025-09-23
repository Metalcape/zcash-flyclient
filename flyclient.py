from zcash_client import ZcashClient
from zcash_mmr import Tree, Node, generate_block_commitments
from ancestry_proof import NodeType, AncestryNode, AncestryProof
import requests
import math
import numpy as np
from secrets import SystemRandom
import json
import equihash

CONF_PATH = "zcash.conf"

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

    peaks: list[int]    # Peaks at chaintip - 1
    peak_heights: list[int] # Heights of peaks at chaintip - 1
    blocks_to_sample: list[int] # list of blocks to sample
    upgrade_names_of_samples: dict[int, str]    # Upgrade of each sampled block
    peak_indices: dict[int, int]    # Map: sampled block -> index of its peak in the chaintip MMR

    ancestry_paths: dict[int, list] # Inclusion paths from the block leaf to the closest peak
    extended_peaks: dict[int, dict] # Map: sampled block -> (Map: network upgrade -> peaks at last block)
    extended_ancestry_paths: dict[int, dict]    # Map: sampled block -> (Map: network upgrade -> inclusion path to closest peak)
    peaks_at_block: dict[int, int]    # Map: sampled block -> peaks at (leaf index - 1) (for chainhistoryroot verification)

    leaves: dict[int, int]

    node_cache: dict[str, dict[int, Node]]
    block_cache: dict[int, dict]

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
        flyclient_activation = self.get_flyclient_activation()
        if override_chain_tip is not None and override_chain_tip > flyclient_activation:
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
        self.blocks_to_sample = blocks_to_sample(flyclient_activation + 1, self.tip_height, c, L)
        self.blocks_to_sample.sort()
        if enable_logging: print(f"Block headers to sample: {self.blocks_to_sample}")

        self.ancestry_paths = dict()
        self.peak_indices = dict()
        self.inclusion_paths = dict()
        self.extended_ancestry_paths = dict()
        self.extended_peaks = dict()
        self.upgrade_names_of_samples = dict()
        self.upgrades_needed = set()
        self.peaks_at_block = dict()
        self.leaves = dict()
        for block_height in self.blocks_to_sample:

            (_, upgrade, activation_height) = self.get_network_upgrade_of_block(block_height)
            
            mmr = Tree([], activation_height)
            if enable_logging: print(f"Processing block {block_height} ({upgrade})")

            # Get the peaks at (leaf - 1) to verify the root
            self.peaks_at_block[block_height] = mmr.peaks_at(block_height - 1)                

            # Compute the ancestry proof indices
            leaf_index = mmr.node_index_of_block(block_height)
            if enable_logging: print(f"Leaf index for block {block_height}: {leaf_index}")
            self.leaves[block_height] = leaf_index
            
            # Calculate the peak index and height of the leaf
            if upgrade == self.upgrade_name:
                last_block = self.tip_height - 1
            else:
                (_, next_activation_height) = self.next_upgrade(upgrade)
                last_block = next_activation_height - 1
            
            (peak_index, peak_h) = self.get_peak_index_and_height(mmr, last_block, block_height)
            if enable_logging: 
                print(f"Peak index for block {block_height}: {peak_index}")
                print(f"Node count at block {block_height}: {mmr.node_count_at(block_height)}")

            # Calculate the path to the chain tip root
            # the leaf is either the leftmost one (before current upgrade) or the one at the sampled block
            if upgrade == self.upgrade_name:
                ancestry_path = self.path_to_root(self.peaks[peak_index], peak_h, leaf_index)
            else:
                ancestry_path = self.path_to_root(self.peaks[0], self.peak_heights[0], 0)
            if enable_logging: print(f"Siblings for block header {block_height}: {ancestry_path}")

            # Check if the ancestry path belongs to the latest upgrade. If not, we must keep
            # climbing the MMR tree(s) until we get to the chaintip root.
            extended_path = dict()
            extended_peaks = dict()
            current_upgrade = upgrade
            current_activation_height = activation_height
            while current_upgrade != self.upgrade_name:
                mmr = Tree([], current_activation_height)

                # Get the last block covered by the network upgrade
                (_, next_activation_height) = self.next_upgrade(current_upgrade)
                last_block = next_activation_height - 1

                # Get peak set, with index and height of the first leaf
                peaks = mmr.peaks_at(last_block)
                peak_heights = mmr.peak_heights_at(last_block)
                
                if current_upgrade == upgrade:
                    path = self.path_to_root(peaks[peak_index], peak_h, leaf_index)
                else:
                    path = self.path_to_root(peaks[0], peak_heights[0], 0)

                # Get the path to the root (the peak is always the first one)
                extended_path[current_upgrade] = path
                extended_peaks[current_upgrade] = peaks
                
                # Go up one network upgrade
                (current_upgrade, current_activation_height) = self.next_upgrade(current_upgrade)

            self.peak_indices[block_height] = peak_index
            self.ancestry_paths[block_height] = ancestry_path
            self.extended_ancestry_paths[block_height] = extended_path
            self.extended_peaks[block_height] = extended_peaks
            self.upgrade_names_of_samples[block_height] = upgrade
            self.upgrades_needed.add(upgrade)
            for k in extended_path.keys():
                self.upgrades_needed.add(k)

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
        peak_h: int = 0
        peaks = tree.peaks_at(rightmost_block)
        peak_heights = tree.peak_heights_at(rightmost_block)
        assert len(peaks) == len(peak_heights)
        leaf = tree.node_index_of_block(block)
        for (j, peak) in enumerate(peaks):
            if leaf < peak:
                peak_h = peak_heights[j]
                break
        return (j, peak_h)
    
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
    
    def get_flyclient_activation(self) -> int:
        for _, v in self.blockchaininfo['upgrades'].items():
            if str(v['name']).lower() == "heartwood":
                return int(v['activationheight'])
            
    def get_branch_id_of_upgrade(self, upgrade: str):
        for k, v in self.blockchaininfo['upgrades'].items():
            if str(v['name']).lower() == upgrade:
                return k
    
    def calculate_total_download_size_bytes(self, cache_nodes = True, derive_parents = True) -> int:
        # version, hashPrevBlock, merkleRoot, blockCommitments, nTime, nBits, nonce, solutionSize, solution
        header_size = 4 + 32 + 32 + 32 + 4 + 4 + 32 + 3 + 1344      
        node_size = 244
        # Tip height, activation height, consensus branch id, upgrade name
        blockchaininfo_size = 4 + 4 + 8 + len(self.upgrade_name)
        authdataroot_size = 32

        nodes: dict[str, list] = dict()
        heights: dict[str, list] = dict()
        nodes_with_heights: dict[str, list[tuple]] = dict()
        for k in self.upgrades_needed:
            nodes[k] = []
            heights[k] = []

        nodes[self.upgrade_name] += self.peaks
        heights[self.upgrade_name] += self.peak_heights
        for k, l in self.inclusion_paths.items():
            nodes[self.upgrade_names_of_samples[k]] += [n[0] for n in l]
            heights[self.upgrade_names_of_samples[k]] += list(range(0, len(l)))
        for k, l in self.ancestry_paths.items():
            nodes[self.upgrade_names_of_samples[k]] += [n[0] for n in l]
            heights[self.upgrade_names_of_samples[k]] += list(range(0, len(l)))
        for _, d in self.extended_ancestry_paths.items():
            for upgrade, l in d.items():
                nodes[upgrade] += [n[0] for n in l]
                heights[upgrade] += list(range(0, len(l)))

        for k in self.upgrades_needed:
            l = list(zip(nodes[k], heights[k]))
            nodes_with_heights[k] = set(l) if cache_nodes else l

        # Try to calculate a parent if both children are present       
        if derive_parents:
            for k in self.upgrades_needed:
                nodes_with_heights[k] = remove_parents(nodes_with_heights[k])

        # Count nodes, cache duplicates if requested
        total_node_count = 0
        for _, s in nodes_with_heights.items():
            if cache_nodes:
                total_node_count += len(set(s))
            else:
                total_node_count += len(s)

        return (
            (header_size + authdataroot_size) * (len(self.blocks_to_sample) + 1)
            + node_size * total_node_count 
            + blockchaininfo_size
        )
    
    def mean_ancestry_length(self) -> float:
        ancestry_lengths = []
        for b in self.blocks_to_sample:
            normal = len(self.ancestry_paths[b])
            ext = 0
            if len(self.extended_ancestry_paths[b]) > 0:
                for _, sub_p in self.extended_ancestry_paths[b].items():
                    ext += len(sub_p)
            else:
                ext = 0
            ancestry_lengths.append(normal + ext)
        return np.mean(ancestry_lengths)
    
    def mean_inclusion_length(self) -> float:
        lengths = [len(p) for _, p in self.inclusion_paths.items()]
        return np.mean(lengths)

    def sample_count(self) -> int:
        return len(self.blocks_to_sample)
    
    def download_header(self, height) -> dict | None:
        if height in self.block_cache.keys():
            return self.block_cache[height]

        try:
            header = self.client.download_header(height, True)
        except requests.exceptions.RequestException as e:
            print(f"Response error when downloading chain tip: {e.response}")
            return None
        
        self.block_cache[height] = header
        return header
    
    def download_auth_data_root(self, height) -> str | None:
        try:
            result = self.client.download_extra_data("getauthdataroot", height)
        except requests.exceptions.RequestException as e:
            print(f"Response error when downloading auth data root: {e.response}")
            return None
        return result["auth_data_root"]
    
    def download_tx_count(self, height) -> dict | None:
        try:
            result = self.client.download_extra_data("getshieldedtxcount", height)
        except requests.exceptions.RequestException as e:
            print(f"Response error when downloading shielded transaction counts: {e.response}")
            return None
        return result

    def download_peaks(self, upgrade_name: str, peak_indices: list) -> list[Node] | None:
        peak_nodes: list[Node] = []
        for i in peak_indices:
            if i in self.node_cache[upgrade_name].keys():
                peak_nodes.append(self.node_cache[upgrade_name][i])
            else:
                try:
                    node_json = client.download_node(upgrade_name, i, True)
                except requests.exceptions.RequestException as e:
                    print(f"Response error when downloading peak node {i}: {e.response}")
                    return None
                node = Node.from_dict(node_json)
                peak_nodes.append(node)
                self.node_cache[upgrade_name][i] = node
        return peak_nodes

    def download_ancestry_proof(self, upgrade_name: str, nodes: list) -> list[AncestryNode] | None:
        siblings: list[AncestryNode] = []
        ancestry_node: AncestryNode = None
        for n, t in nodes:
            if n in self.node_cache[upgrade_name].keys():
                ancestry_node = AncestryNode(n, self.node_cache[upgrade_name][n], t)
                siblings.append(ancestry_node)
            else:
                try:
                    json_obj = client.download_node(upgrade_name, n, True)
                except requests.exceptions.RequestException as e:
                    print(f"Response error when downloading history node {n}: {e.response}")
                    return None
                node = Node.from_dict(json_obj)
                ancestry_node = AncestryNode(n, node, t)
                siblings.append(ancestry_node)
                self.node_cache[upgrade_name][n] = node
        return siblings
    
    def verify_header(self, block_header: dict) -> bool:
        blockcommitments = block_header["blockcommitments"]
        chain_history_root = block_header["chainhistoryroot"]
        height = block_header["height"]
        if blockcommitments != chain_history_root:
            auth_data_root = self.download_auth_data_root(height)
            gen_commitments = generate_block_commitments(chain_history_root, auth_data_root)
            if gen_commitments != blockcommitments:
                return False
        return verify_pow(block_header)
    
    def download_and_verify(self) -> bool:
        self.node_cache = dict()
        for u in self.upgrades_needed:
            self.node_cache[u] = dict()
        self.block_cache = dict()

        # Download and verify tip header
        chaintip_header = self.download_header(self.tip_height)
        if chaintip_header is None: return False
        chaintip_peak_nodes = self.download_peaks(self.upgrade_name, self.peaks)
        if chaintip_peak_nodes is None: return False
        if self.enable_logging:
            print("Chaintip header:")
            print(json.dumps(chaintip_header, indent=1))
        if self.verify_header(chaintip_header) is False:
            print(f"Fatal: header verification failed for chaintip block {block_height}")
            return False
        
        for block_height in self.blocks_to_sample:
            # Download header
            header = self.download_header(block_height)
            if header is None: return False

            if self.enable_logging: 
                print(f"Sampled header at height {block_height}:")
                print(json.dumps(header, indent=1))

            # Verify block PoW and commitment
            if self.verify_header(header) is False:
                print(f"Fatal: header verification failed for block {block_height}")
                return False
            
            current_upgrade = self.upgrade_names_of_samples[block_height]
            current_branch_id = self.get_branch_id_of_upgrade(current_upgrade)
            
            # Use the tip header if at most recent upgrade
            # Otherwise, download the last header for the current upgrade
            if current_upgrade == self.upgrade_name:
                root_header = chaintip_header
            else:
                (_, next_activation_height) = self.next_upgrade(current_upgrade)
                root_header = self.download_header(next_activation_height)
                if root_header is None: return False

                if self.enable_logging: 
                    print("Root node header:")
                    print(json.dumps(root_header, indent=1))
                
                if self.verify_header(root_header) is False:
                    print("Fatal: root node header verification failed")
                    return False

            # TODO: Download extra info needed to calculate the leaf from the block, and generate the leaf
            try:
                leaf_node = Node.from_dict(client.download_node(current_upgrade, self.leaves[block_height], True))
                # tx_info = self.download_tx_count(block_height)
            except requests.exceptions.RequestException as e:
                print(f"Response error when downloading transaction info: {e.response}")
                return False
            
            # Download peaks at sampled block - 1
            peak_nodes_at_sample = self.download_peaks(current_upgrade, self.peaks_at_block[block_height])
            if peak_nodes_at_sample is None: return False

            # Check chain history root for sampled header
            sample_proof = AncestryProof(None, peak_nodes_at_sample, None, None)
            if not sample_proof.verify_root_from_peaks(header["chainhistoryroot"], current_branch_id):
                print(f"Fatal: sample chainhistoryroot verification failed for block at height {block_height}")
                return False

            # Download MMR nodes for ancestry proof
            siblings = self.download_ancestry_proof(self.upgrade_name, self.ancestry_paths[block_height])
            if siblings is None: return False

            # Download leftmost leaf if not at latest upgrade
            if current_upgrade != self.upgrade_name:
                try:
                    leaf_node = Node.from_dict(client.download_node(self.upgrade_name, 0, True))
                except requests.exceptions.RequestException as e:
                    print(f"Response error when downloading rightmost leaf node: {e.response}")
                    return False

            # Recompute chain history root from node chain and peaks
            peak_index = self.peak_indices[block_height] if current_upgrade == self.upgrade_name else 0
            ancestry_proof = AncestryProof(leaf_node, chaintip_peak_nodes, siblings, peak_index)
            if not ancestry_proof.verify_chain_history_root(chaintip_header["chainhistoryroot"], self.branch_id):
                print(f"Fatal: tip chainhistoryroot verification failed for block at height {block_height}")
                return False

            # Handle extended path if needed
            extra_paths = self.extended_ancestry_paths[block_height]
            extra_peaks = self.extended_peaks[block_height]
            while current_upgrade != self.upgrade_name:
                (_, next_activation_height) = self.next_upgrade(current_upgrade)
                root_header = self.download_header(next_activation_height)
                if root_header is None: return False

                if self.verify_header(root_header) is False:
                    print("Fatal: root node header verification failed")
                    return False

                # If we are in the first network upgrade, get the leaf before the block
                # Otherwise, get the leftmost leaf
                if current_upgrade == self.upgrade_names_of_samples[block_height]:
                    leaf_index = self.leaves[block_height]
                    peak_index = self.peak_indices[block_height]
                else:
                    leaf_index = 0
                    peak_index = 0
                try:
                    leaf_node = Node.from_dict(client.download_node(current_upgrade, leaf_index, True))
                except requests.exceptions.RequestException as e:
                    print(f"Response error when downloading leftmost leaf node for upgrade {current_upgrade}: {e.response}")
                    return False
                
                # Download MMR nodes for extended ancestry proof
                siblings = self.download_ancestry_proof(current_upgrade, extra_paths[current_upgrade])
                if siblings is None: return False

                # Download peaks
                if current_upgrade == self.upgrade_name:
                    peak_nodes = self.download_peaks(current_upgrade, self.peaks)
                else:
                    peak_nodes = self.download_peaks(current_upgrade, extra_peaks[current_upgrade])
                if peak_nodes is None: return False

                # Ancestry proof from activation block to upgrade root
                ancestry_proof = AncestryProof(leaf_node, peak_nodes, siblings, peak_index)
                if not ancestry_proof.verify_chain_history_root(root_header["chainhistoryroot"], current_branch_id):
                    print(f"Fatal: extended chainhistoryroot verification failed for block at height {block_height} at upgrade {current_upgrade}")
                    return False
                
                # Go to next network upgrade
                (current_upgrade, _) = self.next_upgrade(current_upgrade)
                current_branch_id = self.get_branch_id_of_upgrade(current_upgrade)

            print(f"Verified block {block_height}")

        return True
    
def remove_parents(nodes_with_height: list[tuple]) -> list[tuple]:
    if nodes_with_height is not None and len(nodes_with_height) > 0:
        for i, h in nodes_with_height:
            if h == 0:
                continue
            left_child = (i - 2**h, h - 1)
            right_child = (i - 1, h - 1)
            if left_child in nodes_with_height and right_child in nodes_with_height:
                nodes_with_height.remove((i, h))
                return remove_parents(nodes_with_height)
    return nodes_with_height

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