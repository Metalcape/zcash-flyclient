# Zcash-flyclient
An implementation in Python of the [flyclient protocol](https://eprint.iacr.org/2019/226) for the Zcash blockchain. It is currently able to verify a flyclient proof provided by a full node. The purpose of this project is to create a proof-of-concept implementation of the flyclient protocol to be later referenced when developing a client intended to run on resource-constrained hardware.

## Usage
To validate a blockchain using a flyclient proof, the script needs to download Merkle Mountain Range (MMR) tree nodes from a Zcash full node (`zebrad`). The current main branch of zebrad does not support downloading arbitrary MMR nodes, so you will need to use [my own fork of `zebrad`](https://github.com/Metalcape/zebra/tree/flyclient) which implements this function via RPC.

> [!WARNING]
> This fork of `zebrad` will update the database version to `27.1.0` to store history nodes (the MMR nodes). This is required for the flyclient protocol to work correctly. Do not point it to your main configuration file if you don't want this to happen to your main database.

Clone and build `zebrad`:
```bash
git clone https://github.com/Metalcape/zebra.git
git checkout flyclient
cargo build --bin zebrad --release
```

Run zebrad (make sure to enable RPC and disable cookie authentication in `zebrad.toml`):
```bash
./target/release/zebrad start
```
The first time you run `zebrad`, you will have to wait for the database upgrade to complete (will take a few minutes on testnet, a few hours on mainnet)

Now clone this repository and install the requirements:
```bash
git clone https://github.com/Metalcape/zcash-flyclient.git
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

To connect to your `zebrad` instance, place a `zcash.conf` file with the following information in the same directory as `zcash_client.py` (username and password are not used for now and can be whatever):
```conf
rpcuser = yourusername
rpcpassword = yourpassword
rpcport = 8232
rpcbind = 127.0.0.1
```

Finally, run `python ./flyclient.py`. The script will print information about the verification process. You can also use `zcash_client.py` to send any RPC commands to `zebrad`:
```bash
python ./zcash_client.py getblockchaininfo
```

## Roadmap

- [x] Verify sampled blocks' rightmost MMR tree leaf using the header's own `chainhistoryroot`
- [x] Verify block headers' proof-of-work
- [x] Use the `authdataroot` field to verify the `blockcommitments` field in the header
- [x] Implement difficulty-based sampling of blocks using the $(c, L)$ parametrized adversary model
- [ ] Non-interactive proof using Fiat-Shamir heuristic


