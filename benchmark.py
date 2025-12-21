from flyclient import FlyclientProof
from zcash_client import ZcashClient, CONF_PATH
from zcash_mmr import Tree
from typing import Literal

import asyncio
import gzip
import pickle
import pandas as pd
import json

ACTIVATION_HEIGHT = 903000
START_HEIGHT = 904000
CHAINTIP = 3104000

HEADERS = "experiments/headers.csv"
HEADERS_BIN = "experiments/headers_bin.csv"
NODES_BIN = "experiments/nodes.csv"

class FlyclientBenchmark(FlyclientProof):
    _OPT_TYPE = Literal['none', 'cache', 'aggregate', 'flyclient_friendly']

    def __init__(self, 
                client: ZcashClient,
                N_a: float = None,
                c: float = 0.5, 
                L: int = 100, 
                override_chain_tip: int | None = None, 
                enable_logging = True, 
                difficulty_aware = False, 
                non_interactive = False):
        
        super(FlyclientBenchmark, self).__init__(client, N_a, c, L, override_chain_tip, enable_logging, difficulty_aware, non_interactive)
    
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

    def nodes_to_download(self, cache_nodes: bool) -> dict[str, list | set]:
        nodes: dict[str, list | dict] = dict()
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
        if cache_nodes:
            nodes = { k: set(v) for k, v in nodes.items() }
        return nodes

    def calculate_proof_size(self, optimization : _OPT_TYPE) -> int:
        match optimization:
            case 'none':
                nodes = self.nodes_to_download(False)
            case 'cache':
                nodes = self.nodes_to_download(True)
            case 'aggregate' | 'flyclient_friendly':
                _, nodes = self.aggregate_proof()
        total_node_count = 0
        for name, l in nodes.items():
            if self.enable_logging:
                print(f"{name}: {l}, length = {len(l)}")
            total_node_count += len(l)
        return total_node_count
    
    @staticmethod
    def to_flyclient_friendly(block_header: dict) -> bytes:
        return b''.join([
            bytes.fromhex(block_header['previousblockhash']),
            bytes.fromhex(block_header['blockcommitments']),
            bytes.fromhex(block_header['bits']),
            int.to_bytes(block_header['time'], 4, 'big')
        ])
    
    async def calculate_compressed_proof_size(self, optimization : _OPT_TYPE, compress_each: bool = True) -> int:        
        blockchaininfo = gzip.compress(pickle.dumps(self.blockchaininfo))
        match optimization:
            case 'none':
                blocks = self.blocks_to_sample
                nodes = self.nodes_to_download(False)
            case 'cache':
                blocks = set(self.blocks_to_sample)
                nodes = self.nodes_to_download(True)
            case 'aggregate' | 'flyclient_friendly':
                blocks, nodes = self.aggregate_proof()
                blocks = set.union(*blocks.values())
        
        if optimization == 'flyclient_friendly':
            block_data = [
                self.to_flyclient_friendly(b)
                for b in await self.client.download_headers_parallel(blocks, True)
            ]
        else:
            block_data = [
                bytes.fromhex(s) for s in await self.client.download_headers_parallel(blocks, False)
            ]

        node_data = {
            upgrade: [
                bytes.fromhex(s) for s in await self.client.download_nodes_parallel(upgrade, ids, False)
            ]
            for upgrade, ids in nodes.items()
        }

        if compress_each:
            comp_block_data = b''.join([gzip.compress(b, 9) for b in block_data])
            comp_node_data = b''.join([b''.join([gzip.compress(n, 9) for n in l]) for _, l in node_data.items()])
        else:
            comp_block_data = gzip.compress(b''.join(block_data))
            comp_node_data = gzip.compress(b''.join([b''.join(l) for l in node_data.values()]))
        
        return len(blockchaininfo) + len(comp_block_data) + len(comp_node_data)
        
    def calculate_total_download_size_bytes(self, optimization : _OPT_TYPE) -> int:
        # Assume hard fork that changes PoW mechanism as such:
        # H_heavy(H_light(B), nBits, timestamp, ChainHistoryRoot) < target
        if optimization == 'flyclient_friendly':
            # H_light(B), nBits, timestamp, ChainHistoryRoot
            header_size = 32 + 4 + 4 + 32
        else:
            # version, hashPrevBlock, merkleRoot, blockCommitments, nTime, nBits, nonce, solutionSize, solution
            header_size = 4 + 32 + 32 + 32 + 4 + 4 + 32 + 3 + 1344
            # Auth data root
            header_size += 32
        
        node_size = 244
        # Tip height, activation height, consensus branch id, upgrade name
        blockchaininfo_size = 4 + 4 + 8 + len(self.upgrade_name)

        if self.enable_logging:
            print(f"Blocks: {self.blocks_to_sample}, length = {len(self.blocks_to_sample)}")

        total_node_count = self.calculate_proof_size(optimization)
            
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
        
        proof = await FlyclientBenchmark.create(client, N_a=50, enable_logging=False, difficulty_aware=True)
        await proof.prefetch()
        # await proof.prefetch_fake_chain(15, [('nu1', 0), ('nu2', 6)], [3, 5, 9])
        print(f"Unoptimized: {proof.calculate_total_download_size_bytes('none')}")
        print(f"Cache nodes: {proof.calculate_total_download_size_bytes('cache')}")
        print(f"Aggregate: {proof.calculate_total_download_size_bytes('aggregate')}")

if __name__ == '__main__':
    asyncio.run(main())