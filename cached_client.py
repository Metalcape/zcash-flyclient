from zcash_client import *
import pandas as pd
import asyncio
import bisect
import json
import sys
import os

BLOCKCHAININFO = "experiments/blockchaininfo.json"
HEADERS = "experiments/headers.csv"
HEADERS_BIN = "experiments/headers_bin.csv"
NODES = "experiments/nodes.csv"
NODES_BIN = "experiments/nodes_bin.csv"
DIFFMAP = "experiments/diffmap.csv"

class CachedClient(ZcashClient):
    _bin_node_cache: dict[str, dict[int, str]] = {}
    _node_cache: dict[str, dict[int, dict]] = {}
    _bin_header_cache: dict[int, str] = {}
    _header_cache: dict[int, dict] = {}
    _diff_cache: dict[int, int] = {}
    _blockchaininfo = {}
    _cache_ready = False

    def __init__(self, user: str, password: str, port: int, host: str, persistent: bool = False):
        super(CachedClient, self).__init__(user, password, port, host, persistent)
        self._build_cache()
       
    @classmethod
    def _build_cache(cls):
        if cls._cache_ready:
            return
        
        print("Reading files for caching...")
        headers = pd.read_csv(HEADERS)
        print("Headers file loaded")
        headers_bin = pd.read_csv(HEADERS_BIN)
        print("Binary headers file loaded")
        nodes = pd.read_csv(NODES)
        print("Nodes file loaded")
        nodes_bin = pd.read_csv(NODES_BIN)
        print("Binary nodes file loaded")
        diffmap = pd.read_csv(DIFFMAP)
        print("Difficulty map loaded")

        print("Building header cache")
        cls._header_cache = {
            k: json.loads(v) for k, v in zip(headers['height'], headers['header'], strict=True)
        }
        print("Building binary header cache")
        cls._bin_header_cache = {
            k: v for k, v in zip(headers_bin['height'], headers_bin['header'], strict=True)
        }

        upgrades = [
            ('heartwood', 903000), 
            ('canopy', 1046400), 
            ('nu5', 1687104), 
            ('nu6', 2726400)
        ]
        
        for name, _ in upgrades:
            df = nodes.query("upgrade==@name")
            print(f"Building node cache for {name}")
            cls._node_cache[name] = {k: json.loads(v) for k, v in zip(df['id'], df['node'], strict=True)}
            df = nodes_bin.query("upgrade==@name")
            print(f"Building binary node cache for {name}")
            cls._bin_node_cache[name] = {k: v for k, v in zip(df['id'], df['node'], strict=True)}
        
        print("Building difficulty cache")
        cls._diff_cache = {
            int(k): int.from_bytes(bytes.fromhex(v), 'big') for k, v in zip(diffmap['height'], diffmap['total_work'])
        }

        with open(BLOCKCHAININFO, 'r') as infile:
            cls._blockchaininfo = json.load(infile)
        
        cls._cache_ready = True
    
    async def download_header(self, height: int, verbose: bool):
        if verbose:
            return self._header_cache[height]
        else:
            return self._bin_header_cache[height]
        
    async def download_node(self, network_upgrade: str, index: int, verbose: bool):
        if verbose:
            return await ZcashClient.download_node(network_upgrade, index, verbose)
        else:
            return self._bin_node_cache[network_upgrade][index]

    async def download_headers_parallel(self, block_heights: list[int], verbose: bool = True):
        if verbose:
            return [self._header_cache[h] for h in block_heights]
        else:
            return [self._bin_header_cache[h] for h in block_heights]
        
    async def download_nodes_parallel(self, network_upgrade: str, nodes: list[int], verbose: bool):
        if verbose:
            return [self._node_cache[network_upgrade][i] for i in nodes]
        else:
            return [self._bin_node_cache[network_upgrade][i] for i in nodes]

    async def get_first_blocks_with_total_work(self, difficulties: list[str]):
        diff_list = sorted([int.from_bytes(bytes.fromhex(d), 'big') for d in difficulties])
        return [
            {'height': self.find_height(d)} for d in diff_list
        ]

    def find_height(self, target_work: int) -> int:
        i = bisect.bisect_left(self._diff_cache.values(), target_work)
        if i >= len(self._diff_cache.values()):
            raise ValueError(f"No height found with work >= {target_work}")
        return self._diff_cache.values()[i]   
    
    async def send_command(self, method: str, params: list[str] = None):
        if method == 'getblockchaininfo':
            return self._blockchaininfo
        else:
            return await ZcashClient.send_command(self, method, params)

async def main():
    cmd = sys.argv[1]
    params = sys.argv[2:len(sys.argv)]
    print(params)

    # async with ZcashClient("flyclient", "", 8232, "127.0.0.1") as client:
    async with CachedClient.from_conf(CONF_PATH) as client:
        result = await client.send_command(cmd, params)

        if result:
            response = json.dumps(result, indent=4)
            print(response)

if __name__ == "__main__":
    asyncio.run(main())