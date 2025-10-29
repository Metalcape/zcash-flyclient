import math
import equihash
from secrets import SystemRandom
import numpy as np

# Equihash parameters
SOL_K = 9
SOL_N = 200

_secure_random = SystemRandom()

# PoW
def verify_pow(block_header: dict) -> bool:
    pow_sol = bytes.fromhex(block_header["solution"])
    pow_header = (
        int(block_header["version"]).to_bytes(4, 'little', signed=True) 
        + bytes.fromhex(block_header["previousblockhash"] )
        + bytes.fromhex(block_header["merkleroot"])
        + bytes.fromhex(block_header["blockcommitments"])
        + int(block_header["time"]).to_bytes(4, 'little', signed=False)
        + bytes.fromhex(block_header["bits"])
        + bytes.fromhex(block_header["nonce"])
    )

    return equihash.verify(SOL_N, SOL_K, pow_header, pow_sol)

def verify_hash(leaf_node: dict, block_header: dict):
        # Assumes that the block hash was already verified
        header_hash = bytes.fromhex(block_header["hash"])
        subtree_commitment = bytes.fromhex(leaf_node["subtree_commitment"])[::-1]
        return header_hash == subtree_commitment
    
# Block sampling
def sample(n: int, min: int, max: int, delta: float):
    if min <= 0 or max <= 0 or max <= min:
        raise ValueError("Invalid boundaries for sampling interval. They must be positive integers with max > min.")

    u_samples = np.array([_secure_random.uniform(0, 1 - delta) for _ in range(n)])
    distrib = 1 - delta**u_samples
    scaled_distrib = min + distrib * (max - min)
    return scaled_distrib

def blocks_to_sample(activation_height: int, chaintip: int, c: float, L: int) -> list[int]:
    # probability of failure is bounded by 2**(-lambda)
    LAMBDA = 50
    # n = chain length
    N = chaintip
    # C = attacker success probability
    # L = estimate of chain difficulty
    # L = delta*n = c**k * n
    # L default is set to the usual size of the non finalized state after sync
    DELTA = L/N
    K = math.log(DELTA, c)

    m = math.ceil(LAMBDA / math.log(1 - (1 / math.log(DELTA, c)), 0.5))

    # Security property
    p_max = (1 - (1/K)) ** m
    assert p_max <= 2 ** (-LAMBDA)

    deterministic = [i for i in range(chaintip - L, chaintip)]
    random = sample(m, activation_height, chaintip - L, DELTA)
    return np.concatenate((random, np.asarray(deterministic, dtype=np.float64))).round().astype(int).tolist()

def difficulty_to_sample(min_work: int, chaintip_total_work: int, c: float, L: int) -> list[int]:
    LAMBDA = 50
    # n equals the total cumulative difficulty in this case
    N = chaintip_total_work
    # L in this case is the fraction of difficulty that is always sampled
    DELTA = L/N
    K = math.log(DELTA, c)

    m = math.ceil(LAMBDA / math.log(1 - (1 / math.log(DELTA, c)), 0.5))

    # Security property
    p_max = (1 - (1/K)) ** m
    assert p_max <= 2 ** (-LAMBDA)

    random = sample(m, min_work, chaintip_total_work - L, DELTA)
    return random.round().astype(int).tolist()