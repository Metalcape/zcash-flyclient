import requests
import configparser
import json
import sys

CONF_PATH = "zcash.conf"
SECTION = "DEFAULT"

class ZcashClient:
    rpcuser = "yourusername"
    rpcpassword = "yourpassword"
    rpcport = 8232
    rpcbind = "127.0.0.1"
    
    def __init__(self, user: str, password: str, port: int, host: str):
        """Initialize the client with manually provided configuration."""
        self.rpcbind = host
        self.rpcport = port
        self.rpcpassword = password
        self.rpcuser = user

    @classmethod
    def from_conf(cls, config_path: str):
        """Initialize the client with a configuration file path."""
        cls = ZcashClient(cls.rpcuser, cls.rpcpassword, cls.rpcport, cls.rpcbind)
        cls.load_conf(config_path)
        return cls

    def load_conf(self, path : str):
        """Load configuration from a zcash.conf file."""
        parser = configparser.ConfigParser()
        with open(path) as file:
            parser.read_string(f"[{SECTION}]\n" + file.read())  # Hack to read a conf file without a section header
        self.rpcuser = parser[SECTION]["rpcuser"]
        self.rpcpassword = parser[SECTION]["rpcpassword"]
        self.rpcport = parser[SECTION]["rpcport"]
        self.rpcbind = parser[SECTION]["rpcbind"]

    def send_command(self, method: str, params: list[str] = None):
        match method:
            case "getblockhash":
                params[0] = int(params[0])
            case "getblock":
                params[0] = str(params[0])
                if len(params) >= 2:
                    params[1] = int(params[1])
            case "getblockheader":
                params[0] = str(params[0])
                if len(params) >= 2:
                    params[1] = False if int(params[1]) == 0 else True
            case "gethistorynode":
                params[0] = str(params[0])
                if len(params) >= 2:
                    params[1] = int(params[1])
                if len(params) >= 3:
                    params[2] = int(params[2])
            case "getauthdataroot" | "getshieldedtxcount" | "gettotalwork" | "getfirstblockwithtotalwork":
                if len(params) > 0:
                    params[0] = str(params[0])

        """Send a JSON-RPC command to a Zcash full node."""
        url = f"http://{self.rpcbind}:{self.rpcport}/"
        headers = {"Content-Type": "application/json"}
        payload = {
            "jsonrpc": "1.0",
            "id": "python-client",
            "method": method,
            "params": params or []
        }
        
        try:
            response = requests.post(url, auth=(self.rpcuser, self.rpcpassword), headers=headers, data=json.dumps(payload))
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with the Zcash node: {e}")
        return response.json()
    
    def download_header(self, height: int, verbose: bool):
        # Get block hash
        hash_response = self.send_command("getblockhash", [f"{height}"])
        if hash_response["error"] is None:
            hash = hash_response['result']
        else:
            raise requests.exceptions.RequestException(response=hash_response["error"])
        
        # Get block header
        response = self.send_command("getblockheader", [f"{hash}", (1 if verbose else 0)])
        if response["error"] is None:
            return response["result"]
        else:
            raise requests.exceptions.RequestException(response=response["error"])
        
    # TODO: Fix zebra to not include entry metadata!
    # TODO: Fix zebra serialization padding zeros
    def download_node(self, network_upgrade: str, index: int, verbose: bool):
        response = self.send_command("gethistorynode", [network_upgrade, index, (1 if verbose else 0)])
        if response["error"] is None:
            return response["result"] if verbose else bytes.fromhex(response["result"])[9:]
        else:
            raise requests.exceptions.RequestException(response=response["error"])
    
    def download_extra_data(self, command: str, height: int):
        response = self.send_command(command, [height])
        if response["error"] is None:
            return response["result"]
        else:
            raise requests.exceptions.RequestException(response=response["error"])

if __name__ == "__main__":
    cmd = sys.argv[1]
    params = sys.argv[2:len(sys.argv)]
    print(params)
    
    client = ZcashClient.from_conf(CONF_PATH)
    # client = ZcashClient("flyclient", "", 8232, "127.0.0.1")
    result = client.send_command(cmd, params)
    
    # Examples
    # result = client.send_command("getblock", [sys.argv[1], 1])
    # result = client.send_command("gethistorynode", [1, "heartwood", 40000])
    # result = client.send_command("getblockheader", [sys.argv[1]])
    # result = client.send_command("getblockchaininfo")
    # result = client.send_command("getexperimentalfeatures")
    # result = client.send_command("z_gettreestate", [sys.argv[1]])
    # result = client.send_command(sys.argv[1], sys.argv[2:])

    if result:
        response = json.dumps(result, indent=4)
        print(response)
