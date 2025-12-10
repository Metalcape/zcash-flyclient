from benchmark import FlyclientBenchmark
from zcash_client import ZcashClient, CONF_PATH
from typing import Literal, get_args
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

async def calc_total_size(client: ZcashClient, row: pd.Series, opt_lv: FlyclientBenchmark._OPT_TYPE, alt_pow: bool = False):
    print(f"Processing sample {row.chaintip, row.c, row.L}, opt_lv={opt_lv}")
    block_list = json.loads(row.samples)
    benchmark = await FlyclientBenchmark.create(client, row.c, row.L, override_chain_tip=row.chaintip, enable_logging=False)
    await benchmark.prefetch(block_list)
    size = benchmark.calculate_total_download_size_bytes(optimization=opt_lv, is_alt_pow=alt_pow)
    print(f"Calculated size for {row.chaintip, row.c, row.L}, opt_lv={opt_lv} => {size / 2**20:.3} MiB")
    return size

async def gen_dataset(file_path: str, samples_file: str, is_alt_pow: bool = False):
    print(f"Generating file: {file_path}")
    samples = pd.read_csv(samples_file)
    semaphore = asyncio.Semaphore(1000)
    pow_type = 'flyclient_friendly' if is_alt_pow else 'default'

    async def bounded_calc(client: ZcashClient, row: pd.Series, opt_lv: FlyclientBenchmark._OPT_TYPE):
        async with semaphore:
            return await calc_total_size(client, row, opt_lv, is_alt_pow)
    
    async with ZcashClient.from_conf(CONF_PATH, persistent=True) as client:
        await client.open()
        df_dict : dict[str, pd.DataFrame] = dict()
        for opt_lv in get_args(FlyclientBenchmark._OPT_TYPE):
            sizes = await asyncio.gather(*[bounded_calc(client, row, opt_lv) for row in samples.itertuples(index=False)])
            df_dict[opt_lv] = pd.DataFrame({
                'chaintip': samples['chaintip'],
                'N_a': samples['N_a'],
                'c': samples['c'],
                'L': samples['L'],
                'difficulty': samples['difficulty'],
                'protocol': samples['protocol'],
                'sample_count': [len(json.loads(s)) for s in samples['samples']],
                'optimization_type': opt_lv,
                'pow_type': pow_type,
                'proof_size': sizes,
                'attack_cost': None
            })
        df = pd.concat(df_dict.values())
        df['attack_cost'] = np.around(C51h * (df['N_a'] / Dh)).astype(int)
        await client.close()

    df.to_csv(file_path, index=False)

async def main():

    jobs = [
        (DATASET_HEIGHT, SAMPLES_HEIGHT),
        (DATASET_DIFF, SAMPLES_DIFFICULTY),
        (DATASET_HEIGHT_NI, SAMPLES_HEIGHT_NI),
        (DATASET_DIFF_NI, SAMPLES_DIFFICULTY_NI),
    ]

    for filepath, sample_path in jobs:
        if not os.path.isfile(filepath):
            await gen_dataset(filepath, sample_path)
    
    dataset = pd.DataFrame(columns=sample_cols)
    df_dict : dict[int, pd.DataFrame] = dict()
    for filepath, _ in jobs:
        if os.path.isfile(filepath):
            df = pd.read_csv(filepath)
            df_dict[filepath] = df
    dataset = pd.concat([df for _, df in df_dict.items()], ignore_index=True)
    if not os.path.isfile(DATASET):
        dataset.to_csv(DATASET, index=False)

if __name__ == '__main__':
    asyncio.run(main())
