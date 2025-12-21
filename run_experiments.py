from benchmark import FlyclientBenchmark
from zcash_client import ZcashClient, CONF_PATH
from cached_client import CachedClient
from typing import get_args
import pandas as pd
import numpy as np
import os
import asyncio
import json

from gen_samples import *

DATASET = "experiments/proof_size_dataset.csv"

DATASET_HEIGHT = 'experiments/bandwidth_cost_height.csv'
DATASET_DIFF = 'experiments/bandwidth_cost_diff.csv'

DATASET_HEIGHT_NI = 'experiments/bandwidth_cost_height_ni.csv'
DATASET_DIFF_NI = 'experiments/bandwidth_cost_diff_ni.csv'

bandwidth_cols = ['height', 'sample_count', 'optimization_type', 'size']
attack_cols = ['height', 'c', 'L', 'size', 'attack_cost']

async def calc_total_size(client: ZcashClient, 
                          row: pd.Series, opt_lv: 
                          FlyclientBenchmark._OPT_TYPE, 
                          compress = False):
    
    print(f"Processing sample ({row.chaintip}, {row.c:.3f}, {row.L}), opt_lv={opt_lv}, compression={compress}")
    block_list = json.loads(row.samples)
    benchmark = await FlyclientBenchmark.create(client, row.c, row.L, override_chain_tip=row.chaintip, enable_logging=False, non_interactive=(row.protocol == 'non_interactive'))
    await benchmark.prefetch(block_list)
    if compress:
        size = await benchmark.calculate_compressed_proof_size(opt_lv, not benchmark.non_interactive)
    else:
        size = benchmark.calculate_total_download_size_bytes(opt_lv)
    print(f"Calculated size for {row.chaintip, row.c, row.L}, opt_lv={opt_lv}, compression={compress} => {size / 2**20:.3} MiB")
    return size

async def gen_dataset(file_path: str, samples_file: str, is_fixed_diff: bool, is_non_interactive: bool):
    print(f"Generating file: {file_path}")
    samples = pd.read_csv(samples_file)
    diff_label = 'constant' if is_fixed_diff else 'variable'
    protocol_label = 'non_interactive' if is_non_interactive else 'interactive'
    samples.query("difficulty==@diff_label and protocol==@protocol_label", inplace=True)

    async with CachedClient.from_conf(CONF_PATH) as client:
        # await client.open()
        df_dict : dict[bool, dict[str, pd.DataFrame]] = dict()
        for compressed in [True, False]:
            df_dict[compressed] = dict()
            for opt_lv in get_args(FlyclientBenchmark._OPT_TYPE):
                # Skip nonsensical combinations
                if opt_lv == 'none' and compressed is True:
                    continue
                chunk_size = 100_000
                chunks = [samples.iloc[i:i+chunk_size] for i in range(0, len(samples), chunk_size)]
                sizes = []
                for c in chunks:
                    sizes += await asyncio.gather(*[
                            calc_total_size(client, row, opt_lv, compressed) 
                            for row in c.itertuples(index=False)
                    ])
                df_dict[compressed][opt_lv] = pd.DataFrame({
                    'chaintip': samples['chaintip'],
                    'N_a': samples['N_a'],
                    'c': samples['c'],
                    'L': samples['L'],
                    'difficulty': samples['difficulty'],
                    'protocol': samples['protocol'],
                    'sample_count': [len(json.loads(s)) for s in samples['samples']],
                    'optimization_type': opt_lv,
                    'compression': compressed,
                    'proof_size': sizes,
                    'attack_cost': None
                })
                print(f"Run completed for {(opt_lv, compressed)}")
        df = pd.concat([pd.concat(dict.values()) for dict in df_dict.values()])
        df['attack_cost'] = np.around(C51h * (df['N_a'] / Dh)).astype(int)
        # await client.close()

    df.to_csv(file_path, index=False)

async def main():
    # FlyclientBenchmark._build_cache()

    jobs = [
        (DATASET_HEIGHT, True, False),
        (DATASET_DIFF, False, False),
        (DATASET_HEIGHT_NI, True, True),
        (DATASET_DIFF_NI, False, True),
    ]

    for filepath, is_fixed_diff, is_non_interactive in jobs:
        if not os.path.isfile(filepath):
            await gen_dataset(filepath, SAMPLES, is_fixed_diff, is_non_interactive)
    
    dataset = pd.DataFrame(columns=sample_cols)
    df_dict : dict[int, pd.DataFrame] = dict()
    for filepath, _, _ in jobs:
        if os.path.isfile(filepath):
            df = pd.read_csv(filepath)
            df_dict[filepath] = df
    dataset = pd.concat([df for _, df in df_dict.items()], ignore_index=True)
    if not os.path.isfile(DATASET):
        dataset.to_csv(DATASET, index=False)

if __name__ == '__main__':
    asyncio.run(main())
