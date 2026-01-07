from zcash_client import ZcashClient, CONF_PATH
from gen_samples import ACTIVATION_HEIGHT, CHAINTIP
from zcash_mmr import Tree
import pandas as pd
import asyncio
import json
import os

HEADERS = "experiments/headers.csv"
HEADERS_BIN = "experiments/headers_bin.csv"
NODES = "experiments/nodes.csv"
NODES_BIN = "experiments/nodes_bin.csv"
STEP = 15000

async def gen_header_cache(file_path: str, verbose: bool):
    print(f"Generating file: {file_path}")
    heights = [h for h in range(ACTIVATION_HEIGHT, CHAINTIP+1)]
    h_ranges = [heights[i:i+STEP] for i in range(0, len(heights), STEP)]
    headers = []
    async with ZcashClient.from_conf(CONF_PATH) as client:
        for r in h_ranges:
            headers += (await client.download_headers_parallel(r, verbose))
            progress = len(headers)
            total = len(heights)
            print(f"{progress} / {total} ({(progress/total * 100):.1f}%)")
        df = pd.DataFrame({
            'height': heights,
            'header': [json.dumps(i) for i in headers] if verbose else headers
        })
    df.to_csv(file_path, index=False)

async def gen_node_cache(file_path: str, verbose: bool):
    print(f"Generating file: {file_path}")
    async with ZcashClient.from_conf(CONF_PATH) as client:
        upgrades = [
            ('heartwood', 903000), 
            ('canopy', 1046400), 
            ('nu5', 1687104), 
            ('nu6', 2726400)
        ]

        last_blocks = [n for _, n in upgrades[1:]] + [CHAINTIP]
        df_dict : dict[str, pd.DataFrame] = {}
        for (name, height), last in zip(upgrades, last_blocks, strict=True):
            print(f"Processing nodes for {name}")
            mmr = Tree([], height)
            last_node = mmr.node_index_of_block(last)
            ids = [i for i in range(0, last_node)]
            id_ranges = [ids[i:i+STEP] for i in range(0, len(ids), STEP)]
            nodes = []
            for r in id_ranges:
                # print(f"Downloading node range {(r[0], r[-1])}")
                nodes += (await client.download_nodes_parallel(name, r, verbose))
                progress = len(nodes)
                total = len(ids)
                print(f"{progress} / {total} ({(progress/total * 100):.1f}%)")
            df_dict[name] = pd.DataFrame({
                'upgrade': name,
                'id': ids,
                'node': [json.dumps(n) for n in nodes] if verbose else nodes
            })
        df = pd.concat(df_dict.values())
    df.to_csv(file_path, index=False)

async def main():
    if not os.path.isfile(HEADERS):
        await gen_header_cache(HEADERS, True)
    if not os.path.isfile(HEADERS_BIN):
        await gen_header_cache(HEADERS_BIN, False)
    if not os.path.isfile(NODES):
        await gen_node_cache(NODES, True)
    if not os.path.isfile(NODES_BIN):
        await gen_node_cache(NODES_BIN, False)

if __name__ == '__main__':
    asyncio.run(main())