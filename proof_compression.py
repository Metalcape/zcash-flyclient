from zcash_client import ZcashClient, CONF_PATH
from demo import FlyclientDemo
import asyncio
import pandas as pd

import lzma
import gzip
import bz2
import zstd

import json
import os

SAMPLES = "experiments/samples_difficulty_ni.csv"
DATASET = "experiments/compression_dataset.csv"
WORKDIR = "temp"

def compress_proof(name: str):
    json_path = name + 'json'
    proof = json.load(json_path)
    formats = {
        "gz": name + '.gz',
        "bz2": name + '.gz',
        "zstd": name + '.zstd',
        "xz": name + '.xz'
    }

    for fmt, path in formats.items():
        if not os.path.exists(path):
            with open(path, 'wb') as outfile:
                match fmt:
                    case "gz":
                        comp_proof = gzip.compress(proof, 9)
                        break
                    case "bz2":
                        comp_proof = bz2.compress(proof, 9)
                        break
                    case "zstd":
                        comp_proof = zstd.compress(proof, 9)
                        break
                    case "xz":
                        comp_proof = lzma.compress(proof, format=lzma.FORMAT_XZ, preset=lzma.PRESET_EXTREME)
                outfile.write(comp_proof)

def compress_multi(proof: str | bytes) -> dict[str, bytes]:
    return {
        "gz": gzip.compress(proof, 9),
        "bz2": bz2.compress(proof, 9),
        "zstd": zstd.compress(proof, 9),
        "xz": lzma.compress(proof, format=lzma.FORMAT_XZ, preset=lzma.PRESET_EXTREME)
    }

async def calc_total_size(client: ZcashClient, row: pd.Series):
    # print(f"Processing sample {row.chaintip, row.c, row.L}")
    block_list = json.loads(row.samples)
    demo = await FlyclientDemo.create(client, row.c, row.L, override_chain_tip=row.chaintip, difficulty_aware=True, non_interactive=True, enable_logging=False)
    await demo.prefetch(block_list)
    proof = json.dumps(await demo.to_dict())
    compress_dict = compress_multi(proof.encode())
    size_dict = { k: len(v) for k, v in compress_dict.items() }
    size_dict["uncompressed"] = len(proof)
    print(f"Calculated size for {row.chaintip, row.c, row.L} => \
    JSON: {size_dict['uncompressed'] / 2**20:.3} MiB,\
    gzip: {size_dict['gz'] / 2**20:.3} MiB,\
    bzip2: {size_dict['bz2'] / 2**20:.3} MiB,\
    zstd: {size_dict['zstd'] / 2**20:.3} MiB,\
    xz: {size_dict['xz'] / 2**20:.3} MiB")
    return size_dict

async def gen_dataset(file_path: str, samples_file: str, is_alt_pow: bool = False):
    print(f"Generating file: {file_path}")
    samples = pd.read_csv(samples_file).query("N_a==50.0").iloc[::8]
    pow_type = 'flyclient_friendly' if is_alt_pow else 'default'
    
    async with ZcashClient.from_conf(CONF_PATH, persistent=True) as client:
        await client.open()
        sizes = [await calc_total_size(client, row) for row in samples.itertuples(index=False)]
        df = pd.DataFrame({
            'chaintip': samples['chaintip'],
            'N_a': samples['N_a'],
            'c': samples['c'],
            'L': samples['L'],
            'difficulty': samples['difficulty'],
            'protocol': samples['protocol'],
            'sample_count': [len(json.loads(s)) for s in samples['samples']],
            'pow_type': pow_type,
            'proof_size_json': [s["uncompressed"] for s in sizes],
            'proof_size_gz': [s["gz"] for s in sizes],
            'proof_size_bz2': [s["bz2"] for s in sizes],
            'proof_size_zstd': [s["zstd"] for s in sizes],
            'proof_size_xz': [s["xz"] for s in sizes],
        })
        await client.close()

    df.to_csv(file_path, index=False)
    
async def main():
    # async with ZcashClient("flyclient", "", 8232, "127.0.0.1") as client:
    async with ZcashClient.from_conf(CONF_PATH, persistent=True) as client:
        await client.open()
        if not os.path.isfile(DATASET):
            await gen_dataset(DATASET, SAMPLES)
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())

