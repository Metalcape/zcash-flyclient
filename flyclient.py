from zcash_client import ZcashClient
from zcash_mmr import *
from sampling import *
from ancestry_proof import path_to_root

from concurrent.futures import ThreadPoolExecutor, as_completed

class FlyclientProof:
    c: float
    L: float
    difficulty_aware: bool
    min_difficulty: int
    max_difficulty: int
    total_difficulty: int

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
    upgrades_needed: set    # Set of upgrades needed to perform the bootstrap
    upgrade_names_of_samples: dict[int, str]    # Upgrade of each sampled block
    
    peak_indices: dict[int, int]    # Map: sampled block -> index of its peak in the chaintip MMR
    ancestry_paths: dict[int, list] # Inclusion paths from the block leaf to the closest peak (at the latest upgrade)
    prev_peaks: dict[str, list[int]]    # Map: network upgrade -> peaks at last block
    extended_ancestry_paths: dict[int, dict[str, list]] # Map: sampled block -> (Map: network upgrade -> inclusion path to closest peak)
    peaks_at_block: dict[int, list[int]]    # Map: sampled block -> peaks at (leaf index - 1) (for chainhistoryroot verification)

    # Fake test chain
    is_fake: bool = False

    def __init__(self, client: ZcashClient, c: float = 0.5, L: int = 100, override_chain_tip: int | None = None, enable_logging = True, difficulty_aware: bool = False):
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

        self.c = c
        max_L = math.trunc(self.c * (self.tip_height - flyclient_activation))
        self.L = L if L < max_L else max_L
        self.difficulty_aware = difficulty_aware

        if self.difficulty_aware:
            mmr = Tree([], self.activation_height)
            self.min_difficulty = int.from_bytes(bytes.fromhex(self.client.download_extra_data("gettotalwork", self.get_flyclient_activation() + 1)["total_work"]), byteorder='big')
            self.max_difficulty = int.from_bytes(bytes.fromhex(self.client.download_node(self.upgrade_name, mmr.insertion_index_of_block(self.tip_height), True)["subtree_total_work"]), byteorder='big')
            self.total_difficulty = int.from_bytes(bytes.fromhex(self.client.download_extra_data("gettotalwork", self.tip_height)["total_work"]), byteorder='big')
        
        if self.activation_height == 0:
            print("Activation height not found.")
        elif enable_logging:
            print(f"Upgrade name: {self.upgrade_name}")
            print(f"Activation height: {self.activation_height}")
    
    def prefetch(self, samples: list[int] = None):
        mmr = Tree([], self.activation_height)
        
        # Get the peaks at the tip
        self.peaks = mmr.peaks_at(self.tip_height - 1)
        self.peak_heights = mmr.peak_heights_at(self.tip_height - 1)
        
        # Choose random blocks, or random cumulative difficulties
        if samples is not None:
            self.blocks_to_sample = samples
        elif self.difficulty_aware:
            self.blocks_to_sample = self.sample_blocks_with_difficulty()
        else:
            self.blocks_to_sample = self.sample_blocks()
        self.blocks_to_sample += self.get_activation_blocks()
        self.blocks_to_sample.sort()

        self.ancestry_paths = dict()
        self.peak_indices = dict()
        self.inclusion_paths = dict()
        self.extended_ancestry_paths = dict()
        self.upgrade_names_of_samples = dict()
        self.upgrades_needed = set()
        self.peaks_at_block = dict()
        self.leaves = dict()
        for block_height in self.blocks_to_sample:

            (_, upgrade, activation_height) = self.get_network_upgrade_of_block(block_height)
            
            mmr = Tree([], activation_height)

            # Get the peaks at (leaf - 1) to verify the root
            if block_height == activation_height:
                _, prev_activation_height = self.prev_upgrade(upgrade)
                prev_mmr = Tree([], prev_activation_height)
                self.peaks_at_block[block_height] = prev_mmr.peaks_at(block_height - 1)
            else:
                self.peaks_at_block[block_height] = mmr.peaks_at(block_height - 1)                

            # Compute the ancestry proof indices
            leaf_index = mmr.node_index_of_block(block_height)
            self.leaves[block_height] = leaf_index
            
            # Calculate the peak index and height of the leaf
            if upgrade == self.upgrade_name:
                last_block = self.tip_height - 1
            else:
                (_, next_activation_height) = self.next_upgrade(upgrade)
                last_block = next_activation_height - 1
            
            (peak_index, peak_h) = mmr.get_peak_index_and_height(last_block, block_height)
            if self.enable_logging: 
                print(f"Peak index for block {block_height}: {peak_index}")
                print(f"Node count at block {block_height}: {mmr.node_count_at(block_height)}")

            # Calculate the path to the chain tip root
            # the leaf is either the leftmost one (before current upgrade) or the one at the sampled block
            if upgrade == self.upgrade_name:
                ancestry_path = path_to_root(self.peaks[peak_index], peak_h, leaf_index)
            else:
                ancestry_path = path_to_root(self.peaks[0], self.peak_heights[0], 0)
            if self.enable_logging: print(f"Siblings for block header {block_height}: {ancestry_path}")

            # Check if the ancestry path belongs to the latest upgrade. If not, we must keep
            # climbing the MMR tree(s) until we get to the chaintip root.
            extended_path = dict()
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
                    path = path_to_root(peaks[peak_index], peak_h, leaf_index)
                else:
                    path = path_to_root(peaks[0], peak_heights[0], 0)

                # Get the path to the root (the peak is always the first one)
                extended_path[current_upgrade] = path
                
                # Go up one network upgrade
                (current_upgrade, current_activation_height) = self.next_upgrade(current_upgrade)

            self.peak_indices[block_height] = peak_index
            self.ancestry_paths[block_height] = ancestry_path
            self.extended_ancestry_paths[block_height] = extended_path
            self.upgrade_names_of_samples[block_height] = upgrade
            self.upgrades_needed.add(upgrade)
            self.upgrades_needed.add(self.upgrade_name)
            for k in extended_path.keys():
                self.upgrades_needed.add(k)
        
        # Save previous peaks for each upgrade needed
        self.prev_peaks = dict()
        for u in self.upgrades_needed.difference(set([self.upgrade_name])):
            activation = self.get_activation_of_upgrade(u)
            _, next_activation = self.next_upgrade(u)
            mmr = Tree([], activation)
            self.prev_peaks[u] = mmr.peaks_at(next_activation - 1)

    def prev_upgrade(self, upgrade_name: str) -> tuple[str, int] | None:
        for k, v in self.blockchaininfo['upgrades'].items():
            if str(v['name']).lower() == upgrade_name:
                keys = list(self.blockchaininfo['upgrades'].keys())
                prev_key_index = keys.index(k) - 1
                if prev_key_index >= 0:
                    prev_key = keys[prev_key_index]
                    return (
                        str(self.blockchaininfo['upgrades'][prev_key]['name']).lower(), 
                        self.blockchaininfo['upgrades'][prev_key]['activationheight']
                    )
                else:
                    return (None, None)
                
        return (None, None)
    
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
            if v['activationheight'] >= height:
                break
            else:
                branch_id = k
        
        return (
            str(branch_id), 
            str(self.blockchaininfo['upgrades'][branch_id]['name']).lower(), 
            self.blockchaininfo['upgrades'][branch_id]['activationheight']
        )
    
    def get_flyclient_activation(self) -> int:
        if self.is_fake:
            return 0
        for _, v in self.blockchaininfo['upgrades'].items():
            if str(v['name']).lower() == "heartwood":
                return int(v['activationheight'])
            
    def get_branch_id_of_upgrade(self, upgrade: str):
        for k, v in self.blockchaininfo['upgrades'].items():
            if str(v['name']).lower() == upgrade:
                return k
    
    def get_activation_of_upgrade(self, upgrade: str) -> int:
        for _, v in self.blockchaininfo['upgrades'].items():
            if str(v['name']).lower() == upgrade:
                return int(v['activationheight'])
            
    def get_activation_blocks(self) -> list[int]:
        flyclient_activation = self.get_flyclient_activation()
        blocks = list()
        for _, v in self.blockchaininfo['upgrades'].items():
            activation = v['activationheight']
            if activation <= flyclient_activation:
                continue
            if activation >= self.tip_height:
                break
            blocks.append(activation)
        return blocks
    
    def sample_blocks(self) -> list[int]:
        return blocks_to_sample(self.get_flyclient_activation() + 1, self.tip_height, self.c, self.L)
    
    def build_difficulty_map(self) -> dict[int, int]:
        difficulty_map: dict[int, int] = dict()
        heights = range(self.get_flyclient_activation() + 1, self.tip_height)

        def fetch_difficulty(height):
            difficulty = int.from_bytes(
                bytes.fromhex(self.client.download_extra_data("gettotalwork", height)["total_work"]), 
                byteorder='big'
            )
            return height, difficulty

        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks
            futures = [executor.submit(fetch_difficulty, i) for i in heights]
            
            # Wait for all to complete and collect results
            for future in as_completed(futures):
                height, difficulty = future.result()
                difficulty_map[height] = difficulty
        
        return difficulty_map

    @staticmethod
    def get_first_block_with_total_work(difficulty_map: dict[int, int], total_work: int) -> int:
        start = min(difficulty_map.keys())
        end = max(difficulty_map.keys())

        while start != end:
            middle = (start + end) // 2
            if difficulty_map[middle] >= total_work:
                end = middle
            else:
                start = middle + 1
        
        return end
    
    def sample_blocks_with_difficulty(self, difficulty_map: dict[int, int] | None = None):     
        # Estimate the dificulty-aware L as L * total work of last block
        max_L = math.trunc(self.c * (self.total_difficulty - 1))
        diff_L = self.L * self.max_difficulty
        diff_L = min(max_L, diff_L)
        difficulty_samples = difficulty_to_sample(self.min_difficulty, self.total_difficulty, self.c, diff_L)

        deterministic = [i for i in range(self.tip_height - self.L, self.tip_height)]
        random = set()
        for d in difficulty_samples:
            if difficulty_map is not None:
                block_index = FlyclientProof.get_first_block_with_total_work(difficulty_map, d)
                random.add(block_index)
            else:
                block_index = self.client.download_extra_data("getfirstblockwithtotalwork", d.to_bytes(32, byteorder='big').hex())
                random.add(block_index['height'])
        
        return list(random) + deterministic
