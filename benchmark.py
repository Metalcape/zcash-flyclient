from flyclient import FlyclientProof
from zcash_client import ZcashClient, CONF_PATH
from zcash_mmr import Tree
from typing import Literal

import asyncio

class FlyclientBenchmark(FlyclientProof):
    _OPT_TYPE = Literal['none', 'cache', 'aggregate']
    _ENC_TYPE = Literal['normal', 'rice_coding', 'flyclient_friendly']

    def __init__(self, 
                client: ZcashClient, 
                c: float = 0.5, 
                L: int = 100, 
                override_chain_tip: int | None = None, 
                enable_logging = True, 
                difficulty_aware = False, 
                non_interactive = False):
        
        super(FlyclientBenchmark, self).__init__(client, c, L, override_chain_tip, enable_logging, difficulty_aware, non_interactive)
    
    def generate_sample_set(self, length: int, with_difficulty : bool = True) -> list[list[int]]:
        samples : list[list[int]] = list()
        for _ in range(length):
            if with_difficulty:
                samples.append(self.sample_blocks_with_difficulty())
            else:
                samples.append(self.sample_blocks())
        return samples
    
    async def prefetch_fake_chain(self, chain_length: int, upgrades: list[tuple[str, int]], samples: list[int]):
        upgrades_dict = dict()
        for i, (u, h) in enumerate(upgrades):
            upgrades_dict[i] = {"name": u, "activationheight": h, "status": "active"}
        self.blockchaininfo['upgrades'] = upgrades_dict
        self.blockchaininfo['blocks'] = chain_length
        self.blockchaininfo['headers'] = chain_length
        self.upgrade_name = upgrades[-1][0]
        self.activation_height = upgrades[-1][1]
        self.tip_height = chain_length
        self.is_fake = True
        await self.prefetch(samples)

    def calculate_proof_size(self, cache_nodes : bool) -> int:
        nodes: dict[str, list] = dict()
        # Populate initial list with nodes from sampled blocks
        mmrs : dict[str, Tree] = dict()
        for u in self.upgrades_needed:
            nodes[u] = list()
            mmrs[u] = Tree([], self.get_activation_of_upgrade(u))
        for b in self.blocks_to_sample:
            upgrade = self.upgrade_names_of_samples[b]
            # Block leaf
            nodes[upgrade].append(mmrs[upgrade].node_index_of_block(b))
            # Ancestry paths to MMR root
            nodes[upgrade] += [n[0] for n in self.ancestry_paths[b]]
            # Other peaks outside ancestry path
            for i in range(len(self.peaks[upgrade])):
                if i != self.peak_indices[b]:
                    nodes[upgrade].append(self.peaks[upgrade][i])

        # Count nodes, cache duplicates if requested
        total_node_count = 0
        for name, l in nodes.items():
            if cache_nodes:
                s = set(l)
                if self.enable_logging:
                    print(f"{name}: {s}, length = {len(s)}")
                total_node_count += len(set(s))
            else:
                if self.enable_logging:
                    print(f"{name}: {l}, length = {len(l)}")
                total_node_count += len(l)
        return total_node_count
    
    def calculate_aggregate_proof_size(self) -> int:
        _, download_set = self.aggregate_proof()
        total_count = 0
        for name, s in download_set.items():
            if self.enable_logging:
                print(f"{name}: {s}, length = {len(s)}")
            total_count += len(s)
        
        return total_count
    
    def calculate_total_download_size_bytes(self, optimization : _OPT_TYPE = 'none', encoding: _ENC_TYPE = 'normal') -> int:
        # Equihash solution: 512 x 21-bit integers -> 1344 bytes total size
        # Using rice coding, each int is on average 14.5 bits long -> average size becomes 928 bits
        equihash_size = 928 if encoding == 'rice_coding' else 1344

        # Assume hard fork that changes PoW mechanism as such:
        # H_heavy(H_light(B), nBits, timestamp, ChainHistoryRoot) < target
        if encoding == 'flyclient_friendly':
            # H_light(B), nBits, timestamp, ChainHistoryRoot
            header_size = 32 + 4 + 4 + 32
        else:
            # version, hashPrevBlock, merkleRoot, blockCommitments, nTime, nBits, nonce, solutionSize, solution
            header_size = 4 + 32 + 32 + 32 + 4 + 4 + 32 + 3 + equihash_size
            # Auth data root
            header_size += 32
        
        node_size = 244
        # Tip height, activation height, consensus branch id, upgrade name
        blockchaininfo_size = 4 + 4 + 8 + len(self.upgrade_name)

        if self.enable_logging:
            print(f"Blocks: {self.blocks_to_sample}, length = {len(self.blocks_to_sample)}")

        match optimization:
            case 'none':
                total_node_count = self.calculate_proof_size(False)
            case 'cache':
                total_node_count = self.calculate_proof_size(True)
            case 'aggregate':
                total_node_count = self.calculate_aggregate_proof_size()
            
        return (
            header_size * (len(self.blocks_to_sample) + 1)
            + node_size * total_node_count 
            + blockchaininfo_size
        )

async def main():
    # async with ZcashClient("flyclient", "", 8232, "127.0.0.1") as client:
    async with ZcashClient.from_conf(CONF_PATH) as client:

        # Example test run on small MMR
        # proof = FlyclientBenchmark(client, enable_logging=True, difficulty_aware=True, override_chain_tip=903809)
        # proof.prefetch([903803, 903806])
        
        proof = await FlyclientBenchmark.create(client, enable_logging=False, difficulty_aware=True)
        await proof.prefetch()
        # await proof.prefetch_fake_chain(15, [('nu1', 0), ('nu2', 6)], [3, 5, 9])
        print(f"Unoptimized: {proof.calculate_total_download_size_bytes('none')}")
        print(f"Cache nodes: {proof.calculate_total_download_size_bytes('cache')}")
        print(f"Aggregate: {proof.calculate_total_download_size_bytes('aggregate')}")

if __name__ == '__main__':
    asyncio.run(main())