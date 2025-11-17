from flyclient import FlyclientProof
from zcash_client import ZcashClient, CONF_PATH
from zcash_mmr import Node, generate_block_commitments
from ancestry_proof import AncestryNode, AncestryProof

import json
import asyncio
import equihash

# Equihash parameters
SOL_K = 9
SOL_N = 200

class FlyclientDemo(FlyclientProof):
    leaves: dict[int, int]
    node_cache: dict[str, dict[int, Node]]
    block_cache: dict[int, dict]

    def __init__(self, 
                client: ZcashClient, 
                c: float = 0.5, 
                L: int = 100, 
                override_chain_tip: int | None = None, 
                enable_logging = True, 
                difficulty_aware = False, 
                non_interactive = False):
        
        super(FlyclientDemo, self).__init__(client, c, L, override_chain_tip, enable_logging, difficulty_aware, non_interactive)
    
    async def download_auth_data_root(self, height) -> str | None:
        result = await self.client.download_extra_data("getauthdataroot", height)
        return result["auth_data_root"]
    
    async def download_tx_count(self, height) -> dict | None:
        result = await self.client.download_extra_data("getshieldedtxcount", height)
        return result

    async def download_peaks(self, upgrade_name: str, peak_indices: list[int]) -> list[Node] | None:
        peak_nodes: dict[int, Node] = {}
        download_list: list[int] = []
        for i in peak_indices:
            if i in self.node_cache[upgrade_name].keys():
                peak_nodes[i] = self.node_cache[upgrade_name][i]
            else:
                download_list.append(i)
        
        nodes_json = await asyncio.gather(*[self.client.download_node(upgrade_name, i, True) for i in download_list])
        # nodes_json = await self.client.download_nodes_parallel(upgrade_name, download_list, True)

        for i, n in zip(download_list, nodes_json, strict=True):
            node = Node.from_dict(n)
            peak_nodes[i] = node
            self.node_cache[upgrade_name][i] = node
        
        return list(peak_nodes.values())

    async def download_ancestry_proof(self, upgrade_name: str, node_index: list) -> list[AncestryNode] | None:
        siblings: list[AncestryNode] = []
        ancestry_node: AncestryNode = None
        download_list = []
        for n, t in node_index:
            if n in self.node_cache[upgrade_name].keys():
                ancestry_node = AncestryNode(n, self.node_cache[upgrade_name][n], t)
                siblings.append(ancestry_node)
            else:
                siblings.append(None)
                download_list.append((n, t))
            
        nodes_json = await asyncio.gather(*[self.client.download_node(upgrade_name, i, True) for i, _ in download_list])
        # nodes_json = await self.client.download_nodes_parallel(upgrade_name, [i for i, _ in download_list], True)
        ancestry_nodes: list[AncestryNode] = []
        for (n, t), data in zip(download_list, nodes_json, strict=True):
            node = Node.from_dict(data)
            ancestry_node = AncestryNode(n, node, t)
            ancestry_nodes.append(ancestry_node)
            self.node_cache[upgrade_name][n] = node

        for i in reversed(range(len(siblings))):
            if siblings[i] is None:
                siblings[i] = ancestry_nodes.pop()

        return siblings
    
    async def download_header(self, height) -> dict | None:
        if height in self.block_cache.keys():
            return self.block_cache[height]
        header = await self.client.download_header(height, True)
        self.block_cache[height] = header
        return header
    
    async def download_headers(self, heights: list[int]) -> dict[int, dict | None]:
        headers : dict[int, dict] = {}
        download_list = []
        for h in heights:
            if h in self.block_cache.keys():
                headers[h] = self.block_cache[h]
            else:
                download_list.append(h)

        downloaded_headers = await self.client.download_headers_parallel(download_list)

        for h, header in zip(download_list, downloaded_headers, strict=True):
            headers[h] = header
            self.block_cache[h] = header
        
        return headers
    
    async def verify_header(self, block_header: dict) -> bool:
        blockcommitments = block_header["blockcommitments"]
        chain_history_root = block_header["chainhistoryroot"]
        height = block_header["height"]
        if blockcommitments != chain_history_root:
            auth_data_root = await self.download_auth_data_root(height)
            gen_commitments = generate_block_commitments(chain_history_root, auth_data_root)
            if gen_commitments != blockcommitments:
                return False
        return verify_pow(block_header)

    async def download_and_verify(self) -> bool:
        self.node_cache = dict()
        for u in self.upgrades_needed:
            self.node_cache[u] = dict()
        self.block_cache = dict()

        # Download and verify tip header
        chaintip_header = await self.download_header(self.tip_height)
        if chaintip_header is None: return False

        if self.enable_logging:
            print("Chaintip header:")
            print(json.dumps(chaintip_header, indent=1))
        if await self.verify_header(chaintip_header) is False:
            print(f"Fatal: header verification failed for chaintip block {self.tip_height}")
            return False
        
        headers = await self.download_headers(self.blocks_to_sample)
        peaks = {
            upgrade : [
                Node.from_dict(d)
                for d in await self.client.download_nodes_parallel(upgrade, nodes, True)
            ]
            for upgrade, nodes in self.peaks.items()
        }
        
        for block_height in self.blocks_to_sample:
            # Download header
            # header = await self.download_header(block_height)
            header = headers[block_height]
            if header is None: return False

            if self.enable_logging: 
                print(f"Sampled header at height {block_height}:")
                print(json.dumps(header, indent=1))

            # Verify block PoW and commitment
            if await self.verify_header(header) is False:
                print(f"Fatal: header verification failed for block {block_height}")
                return False
            
            current_upgrade = self.upgrade_names_of_samples[block_height]
            current_branch_id = self.get_branch_id_of_upgrade(current_upgrade)
            # peak_nodes = await self.download_peaks(current_upgrade, self.peaks[current_upgrade])
            peak_nodes = peaks[current_upgrade]
            
            # Use the tip header if at most recent upgrade
            # Otherwise, download the last header for the current upgrade
            if current_upgrade == self.upgrade_name:
                root_header = chaintip_header
            else:
                (_, next_activation_height) = self.next_upgrade(current_upgrade)
                root_header = await self.download_header(next_activation_height)
                if root_header is None: return False

                if self.enable_logging: 
                    print("Root node header:")
                    print(json.dumps(root_header, indent=1))
                
                if await self.verify_header(root_header) is False:
                    print("Fatal: root node header verification failed at last header of upgrade")
                    return False

            # Download the leaf, and verify that it corresponds to the header using the hash field
            if self.leaves[block_height] in self.node_cache[current_upgrade]:
                node_dict = self.node_cache[current_upgrade][self.leaves[block_height]].to_dict()
            else:
                node_dict = await self.client.download_node(current_upgrade, self.leaves[block_height], True)
            if verify_hash(node_dict, header) == False:
                print(f"Invalid hash for leaf node {self.leaves[block_height]} at block height {block_height}")
                return False
            leaf_node = Node.from_dict(node_dict)
            
            # Download peaks at sampled block - 1
            activation_height = self.get_activation_of_upgrade(current_upgrade)
            if block_height == activation_height:
                prev_upgrade, _ = self.prev_upgrade(current_upgrade)
                branch_id = self.get_branch_id_of_upgrade(prev_upgrade)
                peak_nodes_at_sample = await self.download_peaks(prev_upgrade, self.peaks_at_block[block_height])
            else:
                branch_id = current_branch_id
                peak_nodes_at_sample = await self.download_peaks(current_upgrade, self.peaks_at_block[block_height])
            if peak_nodes_at_sample is None: return False

            # Check chain history root for sampled header
            sample_proof = AncestryProof(None, peak_nodes_at_sample, None, None)
            if not sample_proof.verify_root_from_peaks(header["chainhistoryroot"], branch_id):
                print(f"Fatal: sample chainhistoryroot verification failed for block at height {block_height}")
                return False

            # Download MMR nodes for ancestry proof
            siblings = await self.download_ancestry_proof(self.upgrade_names_of_samples[block_height], self.ancestry_paths[block_height])
            if siblings is None: return False

            # Recompute chain history root from node chain and peaks
            peak_index = self.peak_indices[block_height]
            ancestry_proof = AncestryProof(leaf_node, peak_nodes, siblings, peak_index)
            if not ancestry_proof.verify_chain_history_root(root_header["chainhistoryroot"], current_branch_id):
                print(f"Fatal: tip chainhistoryroot verification failed for block at height {block_height}")
                return False

            print(f"Verified block {block_height}")

        return True
    
    async def to_file(self, path: str):
        proof = {
            'blockchaininfo': {},
            'blocks': {},
            'authdataroot': {},
            'nodes': {}
        }

        proof['blockchaininfo'] = self.blockchaininfo
        
        blocks, download_set = self.aggregate_proof()
        headers = {
            upgrade : await self.download_headers(heights)
            for upgrade, heights in blocks.items()
        }
        
        nodes = {
            upgrade : [
                await self.client.download_nodes_parallel(upgrade, nodes, True)
            ]
            for upgrade, nodes in download_set.items()
        }

        proof["blocks"] = headers
        proof["nodes"] = nodes

        with open(path, 'w') as outfile:
            json.dump(proof, outfile, indent=4)
    
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

async def main():
    # async with ZcashClient("flyclient", "", 8232, "127.0.0.1") as client:
    async with ZcashClient.from_conf(CONF_PATH, persistent=True) as client:
        await client.open()
        proof = await FlyclientDemo.create(client, difficulty_aware=True, non_interactive=True, enable_logging=False)
        await proof.prefetch()
        if await proof.download_and_verify() is True:
            print("Success!")
            await proof.to_file("proof.json")
            print("Proof file written to ./proof.json")
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())