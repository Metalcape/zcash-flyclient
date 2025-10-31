from benchmark import FlyclientBenchmark
from zcash_client import ZcashClient, CONF_PATH
from sampling import difficulty_to_sample
import pandas as pd
import os
import asyncio
import bisect
import math

DIFFMAP = "experiments/diffmap.csv"

SAMPLES_BANDWIDTH_HEIGHT = "experiments/samples_height.csv"
SAMPLES_ATTACK_HEIGHT = "experiments/samples_attack_height.csv"
SAMPLES_BANDWIDTH_DIFFICULTY = "experiments/samples_difficulty.csv"
SAMPLES_ATTACK_DIFFICULTY = "experiments/samples_attack_difficulty.csv"
ITERATIONS = 30

# ACTIVATION_HEIGHT = 903000
START_HEIGHT = 904000
# CHAINTIP = 3504000
CHAINTIP = 3104000
STEP = 25000

Dh = 48 # Zcash hourly number of blocks mined
C_VALUES = [0.2, 0.35, 0.5, 0.65, 0.8]
L_VALUES = [100, Dh * 3, Dh * 4, Dh * 5]
# Estimated cost of a 51% attack on Zcash per hour
C51h = 2624 # USD

sample_cols = ['chaintip', 'c', 'L', 'samples']

class DifficultySampler:
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.work_values = [int(w, 16) for w in df['total_work']]
        self.heights = df['height'].tolist()
    
    async def find_height(self, target_work: int) -> int:
        # target_int = int(target_work, 16)
        i = bisect.bisect_left(self.work_values, target_work)
        if i >= len(self.work_values):
            raise ValueError(f"No height found with work >= {target_work}")
        return self.heights[i]
    
    async def sample(self, chaintip: int, c: float, L: int, min_diff: int,  max_diff: int, total_diff: int):     
        max_L = math.trunc(c * (total_diff - 1))
        diff_L = total_diff - max_diff
        diff_L = min(max_L, diff_L)
        
        difficulty_samples = difficulty_to_sample(min_diff, total_diff, c, diff_L)
        deterministic = [i for i in range(chaintip - L, chaintip)]
        random = await asyncio.gather(*[self.find_height(w) for w in difficulty_samples])
        random.sort()
        return random + deterministic
    
async def gen_diffmap(file_path: int, start_height: int, end_height: int):
    heights = [h for h in range(start_height, end_height)]
    async with ZcashClient.from_conf(CONF_PATH, persistent=True) as client:
        await client.open()
        semaphore = asyncio.Semaphore(STEP / 10)
    
        async def bounded_download(h):
            async with semaphore:
                print(f"Awaiting: {h}")
                response = await client.download_extra_data("gettotalwork", h)
                print(f"Received: {h} => {response}")
                return response
        
        responses = await asyncio.gather(*[bounded_download(h) for h in heights])
        await client.close()
    if not os.path.isfile(file_path):
        df = pd.DataFrame({
            'height': heights,
            'total_work': [r['total_work'] for r in responses]
        })
        df.set_index('height', inplace=True)
        df.sort_index()
        print(df)
        df.to_csv(file_path)

async def gen_samples(file_path: str, var_params: bool, with_diff: bool) -> pd.DataFrame:
    async with ZcashClient.from_conf(CONF_PATH, persistent=True) as client:
        samples = pd.DataFrame(columns=sample_cols)
        params : list[tuple[int, float, int]] = list()
        preset : list[tuple[int, float, int, FlyclientBenchmark]] = list()

        await client.open()
        if var_params:
            for h in range(START_HEIGHT, CHAINTIP, STEP):
                for c in C_VALUES:
                    for l in L_VALUES:
                        params.append((h, c, l))
        else:
            for h in range(START_HEIGHT, CHAINTIP, STEP):
                params.append((h, 0.5, 100))
        
        objects = await asyncio.gather(
            *[FlyclientBenchmark.create(
                client, 
                c, 
                l, 
                enable_logging=False, 
                difficulty_aware=with_diff, 
                override_chain_tip=h
            ) for h, c, l in params])
        preset = [(h, c, l, obj) for (h, c, l), obj in zip(params, objects, strict=True)]

        df_dict : dict[int, pd.DataFrame] = dict()
        if with_diff:
            sampler = DifficultySampler(DIFFMAP)
            for i in range(ITERATIONS):
                print(f"Sampling with difficulty ({i+1}/{ITERATIONS})")
                sample_lists = await asyncio.gather(
                    *[sampler.sample(
                        h,
                        c,
                        l,
                        obj.min_difficulty,
                        obj.max_difficulty,
                        obj.total_difficulty
                    ) for h, c, l, obj in preset])
                new_rows = [
                    {'chaintip': h, 'c': c, 'L': l, 'samples': blocks} 
                    for (h, c, l, _), blocks in zip(preset, sample_lists, strict=True)
                ]
                df_dict[i] = pd.DataFrame(new_rows)
        else:
            for i in range(ITERATIONS):
                print(f"Sampling with height ({i+1}/{ITERATIONS})")
                new_rows = [
                    {'chaintip': h, 'c': c, 'L': l, 'samples': obj.sample_blocks()} for h, c, l, obj in preset
                ]
                df_dict[i] = pd.DataFrame(new_rows)
        samples = pd.concat([df for _, df in df_dict.items()], ignore_index=True)
        samples.sort_index()

        if not os.path.isfile(file_path):
            samples.to_csv(file_path, index=False)
        
        await client.close()
        return samples

async def main():
    if not os.path.isfile(DIFFMAP):
        await gen_diffmap(DIFFMAP, START_HEIGHT, CHAINTIP)

    jobs = [
        (SAMPLES_BANDWIDTH_HEIGHT, False, False),
        (SAMPLES_BANDWIDTH_DIFFICULTY, False, True),
        (SAMPLES_ATTACK_HEIGHT, True, False),
        (SAMPLES_ATTACK_DIFFICULTY, True, True)
    ]

    for filepath, var_params, with_diff in jobs:
        if not os.path.isfile(filepath):
            print(f"Generating file: {filepath}")
            await gen_samples(filepath, var_params, with_diff)

if __name__ == '__main__':
    asyncio.run(main())
