from benchmark import FlyclientBenchmark
from zcash_client import ZcashClient, CONF_PATH
from typing import Literal, get_args
import pandas as pd
import numpy as np
import os
import asyncio
import json

from gen_samples import *

DATASET_BANDWIDTH_HEIGHT = 'experiments/bandwidth_cost_height.csv'
DATASET_BANDWIDTH_DIFF = 'experiments/bandwidth_cost_diff.csv'
DATASET_ATTACK_HEIGHT = 'experiments/attack_cost_height.csv'
DATASET_ATTACK_DIFF = 'experiments/attack_cost_diff.csv'

DATASET_BANDWIDTH_HEIGHT_NI = 'experiments/bandwidth_cost_height_ni.csv'
DATASET_BANDWIDTH_DIFF_NI = 'experiments/bandwidth_cost_diff_ni.csv'
DATASET_ATTACK_HEIGHT_NI = 'experiments/attack_cost_height_ni.csv'
DATASET_ATTACK_DIFF_NI = 'experiments/attack_cost_diff_ni.csv'

DATASET_BANDWIDTH_DIFF_NI_RC = 'experiments/bandwidth_cost_diff_ni_rc.csv'

bandwidth_cols = ['height', 'sample_count', 'optimization_type', 'size']
attack_cols = ['height', 'c', 'L', 'size', 'attack_cost']

__DATASET_TYPE = Literal['bandwidth', 'attack']

async def calc_total_size(client: ZcashClient, row: pd.Series, opt_lv: FlyclientBenchmark._OPT_TYPE, rice_coding: bool = False):
    print(f"Processing sample {row.chaintip, row.c, row.L}, opt_lv={opt_lv}")
    block_list = json.loads(row.samples)
    benchmark = await FlyclientBenchmark.create(client, row.c, row.L, override_chain_tip=row.chaintip, enable_logging=False)
    await benchmark.prefetch(block_list)
    size = benchmark.calculate_total_download_size_bytes(optimization=opt_lv, rice_coding=rice_coding)
    print(f"Calculated size for {row.chaintip, row.c, row.L}, opt_lv={opt_lv} => {size / 2**20:.3} MiB")
    return size

async def gen_dataset(file_path: str, samples_file: str, type: __DATASET_TYPE, rice_coding: bool = False):
    print(f"Generating file: {file_path}")
    samples = pd.read_csv(samples_file)
    semaphore = asyncio.Semaphore(STEP if type=='bandwidth' else 1000)

    async def bounded_calc(client: ZcashClient, row: pd.Series, opt_lv: FlyclientBenchmark._OPT_TYPE):
        async with semaphore:
            return await calc_total_size(client, row, opt_lv, rice_coding)
    
    async with ZcashClient.from_conf(CONF_PATH, persistent=True) as client:
        await client.open()
        match type:
            case 'attack':
                sizes = await asyncio.gather(*[bounded_calc(client, row, 'aggregate') for row in samples.itertuples(index=False)])
                # sizes = [await calc_total_size(client, row, 'aggregate') for row in samples.itertuples(index=False)]
                df = pd.DataFrame({
                    'height': [h for h in samples['chaintip']],
                    'c': samples['c'],
                    'L': samples['L'],
                    'size': sizes,
                    'attack_cost': None
                })
                df['attack_cost'] = np.around(C51h * (df['c'] * df['L'] / Dh)).astype(int)
            case 'bandwidth':
                df_dict : dict[str, pd.DataFrame] = dict()
                for opt_lv in get_args(FlyclientBenchmark._OPT_TYPE):
                    sizes = await asyncio.gather(*[bounded_calc(client, row, opt_lv) for row in samples.itertuples(index=False)])
                    df_dict[opt_lv] = pd.DataFrame({
                        'height': [h for h in samples['chaintip']],
                        'sample_count': [len(json.loads(s)) for s in samples['samples']],
                        'optimization_type': opt_lv,
                        'size': sizes
                    })
                df = pd.concat(df_dict.values())
        await client.close()

    df.to_csv(file_path, index=False)

async def main():

    jobs = [
        (DATASET_BANDWIDTH_HEIGHT, SAMPLES_BANDWIDTH_HEIGHT, 'bandwidth'),
        (DATASET_BANDWIDTH_DIFF, SAMPLES_BANDWIDTH_DIFFICULTY, 'bandwidth'),
        (DATASET_ATTACK_HEIGHT, SAMPLES_ATTACK_HEIGHT, 'attack'),
        (DATASET_ATTACK_DIFF, SAMPLES_ATTACK_DIFFICULTY, 'attack'),

        (DATASET_BANDWIDTH_HEIGHT_NI, SAMPLES_BANDWIDTH_HEIGHT_NI, 'bandwidth'),
        (DATASET_BANDWIDTH_DIFF_NI, SAMPLES_BANDWIDTH_DIFFICULTY_NI, 'bandwidth'),
        (DATASET_ATTACK_HEIGHT_NI, SAMPLES_ATTACK_HEIGHT_NI, 'attack'),
        (DATASET_ATTACK_DIFF_NI, SAMPLES_ATTACK_DIFFICULTY_NI, 'attack')
    ]

    for filepath, sample_path, type in jobs:
        if not os.path.isfile(filepath):
            await gen_dataset(filepath, sample_path, type)
    
    if not os.path.isfile(DATASET_BANDWIDTH_DIFF_NI_RC):
        await gen_dataset(DATASET_BANDWIDTH_DIFF_NI_RC, SAMPLES_BANDWIDTH_DIFFICULTY_NI, 'bandwidth', True)

if __name__ == '__main__':
    asyncio.run(main())
