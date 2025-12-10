import lzma
import gzip
from bitstring import BitArray

def rice_encode(n: int, k = 2, extended = False) -> BitArray:
    if extended:
        n = -2 * n -1 if n < 0 else 2*n
    assert n >= 0
    m = 2**k
    q = n//m
    r = n % m
    bin_str = '1' * q + format(r, '0{}b'.format(k + 1))
    return BitArray(bin=bin_str, length=len(bin_str))

def rice_decode(a: BitArray, k = 2, extended = False) -> int:
    m = 2**k
    q, _ = a.find('0')
    r = a[q:].uint
    i = (q) * m + r
    if extended:
        i = -(i+1)//2 if i%2 else i//2
    return i

def compress_equihash_solution(solution: str) -> str:
    sol_bits = BitArray(hex=solution, length=1344*8)
    sol_parts = [sol_bits[i:i+21] for i in range(0, len(sol_bits), 21)]
    sol_integers = sorted([s.int for s in sol_parts], reverse=True)
    deltas = [sol_integers[i] - sol_integers[i+1] for i in range(0, len(sol_integers), 2)]

    
    enc_integers = [rice_encode(i, 13, True) for i in sol_integers]
    compressed = enc_integers[0].join(enc_integers[1:])
    bytestring = compressed.tobytes()
    return bytestring.hex()

