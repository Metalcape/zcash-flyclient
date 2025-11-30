import math
from secrets import SystemRandom
from randomgen import ChaCha
import numpy as np

class FlyclientSampler:
    a: int
    N: int
    L: int
    c: float

    delta: float
    k: float
    m: int

    _secure_random: SystemRandom | np.random.Generator = None

    def __init__(self, a: int, N: int, L: int, c: float, seed: int = None):
        # probability of failure is bounded by 2**(-lambda)
        LAMBDA = 50
        # a = Flyclient activation height, or minimum cumulative difficulty (difficulty-aware case)
        # N = chain length, or total cumulative difficulty (difficulty-aware case)
        # c = attacker success probability
        # L = amount of blocks (or total difficulty) that is always sampled
        self.a = a
        self.N = N
        self.L = L
        self.c = c
        # L = delta*n = c**k * n
        self.delta = L/N
        self.k = math.log(self.delta, c)

        # Security property (differs in the non interactive case)
        if seed is not None:
            self.m = math.ceil((LAMBDA - math.log(c * N, 0.5)) / math.log(1 - (1 / math.log(self.delta, c)), 0.5))
            p_max = (1 - (1/self.k)) ** self.m
            assert p_max <= (2 ** (-LAMBDA)) / (c * N)
            self._secure_random = np.random.Generator(ChaCha(seed=seed, rounds=8))
        else:
            self.m = math.ceil(LAMBDA / math.log(1 - (1 / math.log(self.delta, c)), 0.5))
            p_max = (1 - (1/self.k)) ** self.m
            assert p_max <= 2 ** (-LAMBDA)
            self._secure_random = SystemRandom()
        
    # Block sampling
    def sample(self, n: int, min: int, max: int):
        if min <= 0 or max <= 0 or max <= min:
            raise ValueError("Invalid boundaries for sampling interval. They must be positive integers with max > min.")

        u_samples = np.array([self._secure_random.uniform(0, 1 - self.delta) for _ in range(n)])
        distrib = 1 - self.delta**u_samples
        scaled_distrib = min + distrib * (max - min)
        return scaled_distrib

    def blocks_to_sample(self) -> list[int]:
        deterministic = [i for i in range(self.N - self.L, self.N)]
        random = self.sample(self.m, self.a, self.N - self.L)
        return np.concatenate((random, np.asarray(deterministic, dtype=np.float64))).round().astype(int).tolist()

    def difficulty_to_sample(self, min_difficulty: int, max_difficulty: int, total_difficulty: int) -> list[int]:
        # Estimate the dificulty-aware L as the total_difficulty - max_difficulty
        # L is the fraction of difficulty that is always sampled
        max_L = math.trunc(self.c * (total_difficulty - 1))
        diff_L = total_difficulty - max_difficulty
        diff_L = min(max_L, diff_L)

        random = self.sample(self.m, min_difficulty, total_difficulty - diff_L)
        return random.round().astype(int).tolist()