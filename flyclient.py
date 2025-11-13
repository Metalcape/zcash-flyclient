from zcash_client import ZcashClient
from zcash_mmr import *
from sampling import FlyclientSampler
from ancestry_proof import path_to_root
import asyncio

class FlyclientProof:
    c: float
    L: float
    difficulty_aware: bool
    non_interactive: bool
    override_chain_tip: int | None
    min_difficulty: int
    max_difficulty: int
    total_difficulty: int

    client: ZcashClient
    sampler: FlyclientSampler
    seed: int | None
    enable_logging: bool
    blockchaininfo: dict
    tip_height: int
    branch_id: str
    activation_height: int
    upgrade_name: str

    peaks: dict[str, list[int]]    # Map: network upgrade -> peaks at last block
    blocks_to_sample: list[int] # list of blocks to sample
    upgrades_needed: set    # Set of upgrades needed to perform the bootstrap
    upgrade_names_of_samples: dict[int, str]    # Upgrade of each sampled block
    
    peak_indices: dict[int, int]    # Map: sampled block -> index of its peak in its upgrade MMR
    ancestry_paths: dict[int, list] # Inclusion paths from the block leaf to the closest peak in its MMR
    peaks_at_block: dict[int, list[int]]    # Map: sampled block -> peaks at (leaf index - 1) (for chainhistoryroot verification)

    # Fake test chain
    is_fake: bool = False

    @classmethod
    async def create(cls, 
                    client: ZcashClient, 
                    c: float = 0.5, 
                    L: int = 100, 
                    override_chain_tip: int | None = None, 
                    enable_logging=True, 
                    difficulty_aware = False, 
                    non_interactive = False):
        
        instance = cls(client, c, L, override_chain_tip, enable_logging, difficulty_aware, non_interactive)
        await instance.initialize()
        return instance

    def __init__(self, 
                client: ZcashClient, 
                c: float = 0.5, 
                L: int = 100, 
                override_chain_tip: int | None = None, 
                enable_logging = True, 
                difficulty_aware = False, 
                non_interactive = False):
        
        self.client = client
        self.enable_logging = enable_logging
        self.override_chain_tip = override_chain_tip
        self.difficulty_aware = difficulty_aware
        self.non_interactive = non_interactive

        # Check that c and L are valid
        if c <= 0 or c >= 1:
            print("c should be between 0 and 1. Falling back to c = 0.5")
            c = 0.5
        if L <= 0:
            print("L should be positive. Falling back to L = 100")
            L = 100
        self.c = c
        self.L = L

    async def initialize(self):
        # Get the tip and activation height
        response = await self.client.send_command("getblockchaininfo")
        self.blockchaininfo: dict = response["result"]
        flyclient_activation = self.get_flyclient_activation()
        if self.override_chain_tip is not None and self.override_chain_tip > flyclient_activation:
            self.tip_height = int(self.override_chain_tip)
        else:
            self.tip_height = int(self.blockchaininfo["blocks"]) - 100
        
        (self.branch_id, self.upgrade_name, self.activation_height) = self.get_network_upgrade_of_block(self.tip_height)

        max_L = math.trunc(self.c * (self.tip_height - flyclient_activation))
        self.L = min(self.L, max_L)

        if self.difficulty_aware:
            # We sample blocks statistically in the difficulty interval [min, max]
            min_diff_response, max_diff_response, total_diff_response = await asyncio.gather(
                self.client.download_extra_data("gettotalwork", self.get_flyclient_activation() + 1),
                self.client.download_extra_data("gettotalwork", self.tip_height - self.L),
                self.client.download_extra_data("gettotalwork", self.tip_height)
            )
            
            self.min_difficulty = int.from_bytes(bytes.fromhex(min_diff_response["total_work"]), byteorder='big')
            self.max_difficulty = int.from_bytes(bytes.fromhex(max_diff_response["total_work"]), byteorder='big')
            self.total_difficulty = int.from_bytes(bytes.fromhex(total_diff_response["total_work"]), byteorder='big')

            # Estimate the dificulty-aware L as the total_difficulty - max_difficulty
            # L is the fraction of difficulty that is always sampled
            max_L = math.trunc(self.c * (self.total_difficulty - 1))
            diff_L = self.total_difficulty - self.max_difficulty
            diff_L = min(max_L, diff_L)

            a = self.min_difficulty
            N = self.total_difficulty
            L = diff_L
            
        else:
            a = flyclient_activation
            N = self.tip_height
            L = self.L
        
        if self.non_interactive:
            tip_block = await self.client.download_header(self.tip_height, True)
            seed = int.from_bytes(bytes.fromhex(tip_block['hash']), byteorder='big')
            self.sampler = FlyclientSampler(a, N, L, self.c, seed)
            self.seed = seed
        else:
            self.sampler = FlyclientSampler(a, N, L, self.c)
            self.seed = None
        
        if self.activation_height == 0:
            print("Activation height not found.")
        elif self.enable_logging:
            print(f"Upgrade name: {self.upgrade_name}")
            print(f"Activation height: {self.activation_height}")
    
    async def prefetch(self, samples: list[int] = None):
        mmr = Tree([], self.activation_height)
        
        # Get the peaks at the tip
        self.peaks = dict()
        self.peaks[self.upgrade_name] = mmr.peaks_at(self.tip_height - 1)
        
        # Choose random blocks, or random cumulative difficulties
        if samples is not None:
            self.blocks_to_sample = samples
        elif self.difficulty_aware:
            self.blocks_to_sample = await self.sample_blocks_with_difficulty()
        else:
            self.blocks_to_sample = self.sample_blocks()
        self.blocks_to_sample += self.get_activation_blocks(self.blocks_to_sample[0])
        self.blocks_to_sample.sort()

        self.ancestry_paths = dict()
        self.peak_indices = dict()
        self.upgrade_names_of_samples = dict()
        self.upgrades_needed = set()
        self.peaks_at_block = dict()
        self.prev_peaks = dict()
        self.leaves = dict()

        self.upgrades_needed.add(self.upgrade_name)

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
            
            # Save the upgrade peaks if we have not yet done so
            if upgrade not in self.peaks.keys():
                self.peaks[upgrade] = mmr.peaks_at(last_block)
            
            (peak_index, peak_h) = mmr.get_peak_index_and_height(last_block, block_height)
            if self.enable_logging: 
                print(f"Peak index for block {block_height}: {peak_index}")
                print(f"Node count at block {block_height}: {mmr.node_count_at(block_height)}")

            # Calculate the path to the MMR root
            ancestry_path = path_to_root(self.peaks[upgrade][peak_index], peak_h, leaf_index)
            if self.enable_logging: print(f"Siblings for block header {block_height}: {ancestry_path}")

            self.peak_indices[block_height] = peak_index
            self.ancestry_paths[block_height] = ancestry_path
            self.upgrade_names_of_samples[block_height] = upgrade
            self.upgrades_needed.add(upgrade)
    
    def aggregate_proof(self) -> tuple[dict[str, set], dict[str, set]]:
        blocks : dict[str, set] = dict()
        download_set : dict[str, set] = dict()

        for u in self.upgrades_needed:
            blocks[u] = set()

        for b in self.blocks_to_sample:
            blocks[self.upgrade_names_of_samples[b]].add(b)
        
        for u in self.upgrades_needed:
            if u == self.upgrade_name:
                mmr = Tree([], self.activation_height)
                download_set[u] = mmr.get_min_size_proof(blocks[u], self.tip_height - 1)
            else:
                _, next_activation_height = self.next_upgrade(u)
                mmr = Tree([], self.get_activation_of_upgrade(u))
                download_set[u] = mmr.get_min_size_proof(blocks[u], next_activation_height - 1)
        
        return (blocks, download_set)

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
            if v['activationheight'] > height:
                break
            elif v['activationheight'] == height:
                branch_id = k
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
            
    def get_activation_blocks(self, first_sampled_block: int) -> list[int]:
        _, first_upgrade, _ = self.get_network_upgrade_of_block(first_sampled_block)
        _, second_activation = self.next_upgrade(first_upgrade)
        if second_activation is None:
            return []
        blocks = list()
        for _, v in self.blockchaininfo['upgrades'].items():
            activation = v['activationheight']
            if activation < second_activation:
                continue
            if activation >= self.tip_height:
                break
            blocks.append(activation)
        return blocks
    
    def sample_blocks(self) -> list[int]:
        if self.sampler is not None:
            return self.sampler.blocks_to_sample()
    
    async def sample_blocks_with_difficulty(self):     
        if self.sampler is None:
            return None
        else:
            difficulty_samples = self.sampler.difficulty_to_sample()

        deterministic = [i for i in range(self.tip_height - self.L, self.tip_height)]
        random = set()

        # Process difficulty queries in parallel
        # requests = [self.client.download_extra_data("getfirstblockwithtotalwork", d.to_bytes(32, byteorder='big').hex()) for d in difficulty_samples]
        # responses = await asyncio.gather(*requests)
        difficulty_strings = [d.to_bytes(32, byteorder='big').hex() for d in difficulty_samples]
        responses = await self.client.get_first_blocks_with_total_work(difficulty_strings)

        for r in responses:
            random.add(r['height'])
        
        return list(random) + deterministic
