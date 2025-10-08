from flyclient import FlyclientProof
from zcash_client import ZcashClient
from zcash_mmr import Node, generate_block_commitments
from sampling import *
from ancestry_proof import AncestryNode, AncestryProof

import json
import requests

CONF_PATH = "zcash.conf"

class FlyclientDemo(FlyclientProof):
    leaves: dict[int, int]
    node_cache: dict[str, dict[int, Node]]
    block_cache: dict[int, dict]

    def __init__(self, client: ZcashClient, c: float = 0.5, L: int = 100, override_chain_tip: int | None = None, enable_logging = True, difficulty_aware: bool = False):
        super(FlyclientDemo, self).__init__(client, c, L, override_chain_tip, enable_logging, difficulty_aware)
        super(FlyclientDemo, self).prefetch()
    
    def download_auth_data_root(self, height) -> str | None:
        try:
            result = self.client.download_extra_data("getauthdataroot", height)
        except requests.exceptions.RequestException as e:
            print(f"Response error when downloading auth data root: {e.response}")
            return None
        return result["auth_data_root"]
    
    def download_tx_count(self, height) -> dict | None:
        try:
            result = self.client.download_extra_data("getshieldedtxcount", height)
        except requests.exceptions.RequestException as e:
            print(f"Response error when downloading shielded transaction counts: {e.response}")
            return None
        return result

    def download_peaks(self, upgrade_name: str, peak_indices: list) -> list[Node] | None:
        peak_nodes: list[Node] = []
        for i in peak_indices:
            if i in self.node_cache[upgrade_name].keys():
                peak_nodes.append(self.node_cache[upgrade_name][i])
            else:
                try:
                    node_json = self.client.download_node(upgrade_name, i, True)
                except requests.exceptions.RequestException as e:
                    print(f"Response error when downloading peak node {i}: {e.response}")
                    return None
                node = Node.from_dict(node_json)
                peak_nodes.append(node)
                self.node_cache[upgrade_name][i] = node
        return peak_nodes

    def download_ancestry_proof(self, upgrade_name: str, nodes: list) -> list[AncestryNode] | None:
        siblings: list[AncestryNode] = []
        ancestry_node: AncestryNode = None
        for n, t in nodes:
            if n in self.node_cache[upgrade_name].keys():
                ancestry_node = AncestryNode(n, self.node_cache[upgrade_name][n], t)
                siblings.append(ancestry_node)
            else:
                try:
                    json_obj = self.client.download_node(upgrade_name, n, True)
                except requests.exceptions.RequestException as e:
                    print(f"Response error when downloading history node {n}: {e.response}")
                    return None
                node = Node.from_dict(json_obj)
                ancestry_node = AncestryNode(n, node, t)
                siblings.append(ancestry_node)
                self.node_cache[upgrade_name][n] = node
        return siblings
    
    def download_header(self, height) -> dict | None:
        if height in self.block_cache.keys():
            return self.block_cache[height]

        try:
            header = self.client.download_header(height, True)
        except requests.exceptions.RequestException as e:
            print(f"Response error when downloading chain tip: {e.response}")
            return None
        
        self.block_cache[height] = header
        return header
    
    def verify_header(self, block_header: dict) -> bool:
        blockcommitments = block_header["blockcommitments"]
        chain_history_root = block_header["chainhistoryroot"]
        height = block_header["height"]
        if blockcommitments != chain_history_root:
            auth_data_root = self.download_auth_data_root(height)
            gen_commitments = generate_block_commitments(chain_history_root, auth_data_root)
            if gen_commitments != blockcommitments:
                return False
        return verify_pow(block_header)

    def download_and_verify(self) -> bool:
        self.node_cache = dict()
        for u in self.upgrades_needed:
            self.node_cache[u] = dict()
        self.block_cache = dict()

        # Download and verify tip header
        chaintip_header = self.download_header(self.tip_height)
        if chaintip_header is None: return False
        chaintip_peak_nodes = self.download_peaks(self.upgrade_name, self.peaks)
        if chaintip_peak_nodes is None: return False
        if self.enable_logging:
            print("Chaintip header:")
            print(json.dumps(chaintip_header, indent=1))
        if self.verify_header(chaintip_header) is False:
            print(f"Fatal: header verification failed for chaintip block {block_height}")
            return False
        
        for block_height in self.blocks_to_sample:
            # Download header
            header = self.download_header(block_height)
            if header is None: return False

            if self.enable_logging: 
                print(f"Sampled header at height {block_height}:")
                print(json.dumps(header, indent=1))

            # Verify block PoW and commitment
            if self.verify_header(header) is False:
                print(f"Fatal: header verification failed for block {block_height}")
                return False
            
            current_upgrade = self.upgrade_names_of_samples[block_height]
            current_branch_id = self.get_branch_id_of_upgrade(current_upgrade)
            
            # Use the tip header if at most recent upgrade
            # Otherwise, download the last header for the current upgrade
            if current_upgrade == self.upgrade_name:
                root_header = chaintip_header
            else:
                (_, next_activation_height) = self.next_upgrade(current_upgrade)
                root_header = self.download_header(next_activation_height)
                if root_header is None: return False

                if self.enable_logging: 
                    print("Root node header:")
                    print(json.dumps(root_header, indent=1))
                
                if self.verify_header(root_header) is False:
                    print("Fatal: root node header verification failed")
                    return False

            # Download the leaf, and verify that it corresponds to the header using the hash field
            try:
                node_dict = self.client.download_node(current_upgrade, self.leaves[block_height], True)
                if verify_hash(node_dict, header) == False:
                    print(f"Invalid hash for leaf node {self.leaves[block_height]} at block height {block_height}")
                    return False
                leaf_node = Node.from_dict(node_dict)
            except requests.exceptions.RequestException as e:
                print(f"Response error when downloading transaction info: {e.response}")
                return False
            
            # Download peaks at sampled block - 1
            peak_nodes_at_sample = self.download_peaks(current_upgrade, self.peaks_at_block[block_height])
            if peak_nodes_at_sample is None: return False

            # Check chain history root for sampled header
            sample_proof = AncestryProof(None, peak_nodes_at_sample, None, None)
            if not sample_proof.verify_root_from_peaks(header["chainhistoryroot"], current_branch_id):
                print(f"Fatal: sample chainhistoryroot verification failed for block at height {block_height}")
                return False

            # Download MMR nodes for ancestry proof
            siblings = self.download_ancestry_proof(self.upgrade_name, self.ancestry_paths[block_height])
            if siblings is None: return False

            # Download leftmost leaf if not at latest upgrade
            if current_upgrade != self.upgrade_name:
                try:
                    leaf_node = Node.from_dict(self.client.download_node(self.upgrade_name, 0, True))
                except requests.exceptions.RequestException as e:
                    print(f"Response error when downloading rightmost leaf node: {e.response}")
                    return False

            # Recompute chain history root from node chain and peaks
            peak_index = self.peak_indices[block_height] if current_upgrade == self.upgrade_name else 0
            ancestry_proof = AncestryProof(leaf_node, chaintip_peak_nodes, siblings, peak_index)
            if not ancestry_proof.verify_chain_history_root(chaintip_header["chainhistoryroot"], self.branch_id):
                print(f"Fatal: tip chainhistoryroot verification failed for block at height {block_height}")
                return False

            # Handle extended path if needed
            extra_paths = self.extended_ancestry_paths[block_height]
            extra_peaks = self.extended_peaks[block_height]
            while current_upgrade != self.upgrade_name:
                (_, next_activation_height) = self.next_upgrade(current_upgrade)
                root_header = self.download_header(next_activation_height)
                if root_header is None: return False

                if self.verify_header(root_header) is False:
                    print("Fatal: root node header verification failed")
                    return False

                # If we are in the first network upgrade, get the leaf before the block
                # Otherwise, get the leftmost leaf
                if current_upgrade == self.upgrade_names_of_samples[block_height]:
                    leaf_index = self.leaves[block_height]
                    peak_index = self.peak_indices[block_height]
                else:
                    leaf_index = 0
                    peak_index = 0
                try:
                    leaf_node = Node.from_dict(self.client.download_node(current_upgrade, leaf_index, True))
                except requests.exceptions.RequestException as e:
                    print(f"Response error when downloading leftmost leaf node for upgrade {current_upgrade}: {e.response}")
                    return False
                
                # Download MMR nodes for extended ancestry proof
                siblings = self.download_ancestry_proof(current_upgrade, extra_paths[current_upgrade])
                if siblings is None: return False

                # Download peaks
                if current_upgrade == self.upgrade_name:
                    peak_nodes = self.download_peaks(current_upgrade, self.peaks)
                else:
                    peak_nodes = self.download_peaks(current_upgrade, extra_peaks[current_upgrade])
                if peak_nodes is None: return False

                # Ancestry proof from activation block to upgrade root
                ancestry_proof = AncestryProof(leaf_node, peak_nodes, siblings, peak_index)
                if not ancestry_proof.verify_chain_history_root(root_header["chainhistoryroot"], current_branch_id):
                    print(f"Fatal: extended chainhistoryroot verification failed for block at height {block_height} at upgrade {current_upgrade}")
                    return False
                
                # Go to next network upgrade
                (current_upgrade, _) = self.next_upgrade(current_upgrade)
                current_branch_id = self.get_branch_id_of_upgrade(current_upgrade)

            print(f"Verified block {block_height}")

        return True
    
if __name__ == "__main__":
    client = ZcashClient.from_conf(CONF_PATH)
    # client = ZcashClient("flyclient", "", 8232, "127.0.0.1")
    
    proof = FlyclientDemo(client, difficulty_aware=True)
    if proof.download_and_verify() is True:
        print("Success!")