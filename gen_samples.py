from benchmark import FlyclientBenchmark
from zcash_client import ZcashClient, CONF_PATH
from sampling import FlyclientSampler
from parameter_optimizer import find_opt_c, opt_eq, opt_eq_ni
import pandas as pd
import os
import asyncio
import bisect
import math

DIFFMAP = "experiments/diffmap.csv"
SAMPLES = "experiments/samples.csv"

SAMPLES_HEIGHT = "experiments/samples_height.csv"
SAMPLES_DIFFICULTY = "experiments/samples_difficulty.csv"
SAMPLES_HEIGHT_NI = "experiments/samples_height_ni.csv"
SAMPLES_DIFFICULTY_NI = "experiments/samples_difficulty_ni.csv"

ITERATIONS = 30

ACTIVATION_HEIGHT = 903000
START_HEIGHT = 904000
# CHAINTIP = 3504000
CHAINTIP = 3104000
STEP = 25000

Dh = 48 # Zcash hourly number of blocks mined
# C_VALUES = [0.2, 0.35, 0.5, 0.65, 0.8]
# L_VALUES = [100, Dh * 3, Dh * 4, Dh * 5]
N_A_VALUES = [25.0, 50.0, 100.0, 500.0]
# Estimated cost of a 51% attack on Zcash per hour
C51h = 2624 # USD

sample_cols = ['chaintip', 'N_a', 'c', 'L', 'samples', 'difficulty', 'protocol']

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
    
    async def sample(self, 
                    chaintip: int, 
                    c: float,
                    L: int,
                    min_diff: int,  
                    max_diff: int, 
                    total_diff: int, 
                    seed: int | None = None):
        
        # c = await find_opt_c(opt_eq_ni, 50, chaintip, N_a)
        # L = math.ceil(N_a / c)
           
        max_L = math.trunc(c * (chaintip - ACTIVATION_HEIGHT))
        L = min(L, max_L)

        sampler = FlyclientSampler(ACTIVATION_HEIGHT, chaintip, L, c, seed)
        difficulty_samples = sampler.difficulty_to_sample(min_diff, max_diff, total_diff)
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

async def gen_samples(file_path: str, with_diff: bool, non_interactive: bool) -> pd.DataFrame:
    async with ZcashClient.from_conf(CONF_PATH, persistent=True) as client:
        protocol = "non_interactive" if non_interactive else "interactive"
        samples = pd.DataFrame(columns=sample_cols)
        params : list[tuple[int, float, int]] = list()
        preset : list[tuple[int, float, FlyclientBenchmark]] = list()

        await client.open()
        for h in range(START_HEIGHT, CHAINTIP, STEP):
            for N_a in N_A_VALUES:
                params.append((h, N_a))
        
        objects = await asyncio.gather(
            *[FlyclientBenchmark.create(
                client, 
                N_a=N_a, 
                enable_logging=False, 
                difficulty_aware=with_diff, 
                override_chain_tip=h,
                non_interactive=non_interactive
            ) for h, N_a in params])
        preset = [(h, N_a, obj) for (h, N_a), obj in zip(params, objects, strict=True)]

        df_dict : dict[int, pd.DataFrame] = dict()
        if with_diff:
            sampler = DifficultySampler(DIFFMAP)
            for i in range(ITERATIONS):
                print(f"Sampling with difficulty ({i+1}/{ITERATIONS})")
                sample_lists = await asyncio.gather(
                    *[sampler.sample(
                        h,
                        obj.c,
                        obj.L,
                        obj.min_difficulty,
                        obj.max_difficulty,
                        obj.total_difficulty,
                        obj.seed
                    ) for (h, _, obj) in preset])
                new_rows = [
                    {
                        'chaintip': h, 
                        'N_a': N_a, 
                        'c': obj.c, 
                        'L': obj.L, 
                        'samples': blocks, 
                        'difficulty': "variable", 
                        'protocol': protocol
                    } for (h, N_a, obj), blocks in zip(preset, sample_lists, strict=True)
                ]
                df_dict[i] = pd.DataFrame(new_rows)
        else:
            for i in range(ITERATIONS):
                print(f"Sampling with height ({i+1}/{ITERATIONS})")
                new_rows = [
                    {
                        'chaintip': h, 
                        'N_a': N_a, 
                        'c': obj.c, 
                        'L': obj.L, 
                        'samples': obj.sample_blocks(), 
                        'difficulty': "constant", 
                        'protocol': protocol
                    } for h, N_a, obj in preset
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
        (SAMPLES_HEIGHT, False, False),
        (SAMPLES_DIFFICULTY, True, False),
        (SAMPLES_HEIGHT_NI, False, True),
        (SAMPLES_DIFFICULTY_NI, True, True),
    ]

    for filepath, with_diff, ni in jobs:
        if not os.path.isfile(filepath):
            print(f"Generating file: {filepath}")
            await gen_samples(filepath, with_diff, ni)
    
    samples = pd.DataFrame(columns=sample_cols)
    df_dict : dict[int, pd.DataFrame] = dict()
    for filepath, _, _ in jobs:
        if os.path.isfile(filepath):
            df = pd.read_csv(filepath)
            df_dict[filepath] = df
    samples = pd.concat([df for _, df in df_dict.items()], ignore_index=True)
    if not os.path.isfile(SAMPLES):
        samples.to_csv(SAMPLES, index=False)

if __name__ == '__main__':
    asyncio.run(main())
