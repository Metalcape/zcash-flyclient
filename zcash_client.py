import aiohttp
import asyncio
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

    persistent: bool = False
    
    def __init__(self, user: str, password: str, port: int, host: str, persistent: bool = False):
        """Initialize the client with manually provided configuration."""
        self.rpcbind = host
        self.rpcport = port
        self.rpcpassword = password
        self.rpcuser = user
        self._session = None
        self.persistent = persistent

    @classmethod
    def from_conf(cls, config_path: str, persistent: bool = False):
        """Initialize the client with a configuration file path."""
        instance = cls(cls.rpcuser, cls.rpcpassword, cls.rpcport, cls.rpcbind, persistent)
        instance.load_conf(config_path)
        return instance

    def load_conf(self, path: str):
        """Load configuration from a zcash.conf file."""
        parser = configparser.ConfigParser()
        with open(path) as file:
            parser.read_string(f"[{SECTION}]\n" + file.read())
        self.rpcuser = parser[SECTION]["rpcuser"]
        self.rpcpassword = parser[SECTION]["rpcpassword"]
        self.rpcport = parser[SECTION]["rpcport"]
        self.rpcbind = parser[SECTION]["rpcbind"]

    async def __aenter__(self):
        if not self.persistent:
            await self.open()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not self.persistent:
            await self.close()

    async def open(self):
        """Open the aiohttp session."""
        if self._session is None:
            auth = aiohttp.BasicAuth(self.rpcuser, self.rpcpassword)
            self._session = aiohttp.ClientSession(auth=auth)

    async def close(self):
        """Close the aiohttp session."""
        if self._session:
            await self._session.close()
            self._session = None

    def _prepare_params(self, method: str, params: list[str] = None):
        """Prepare parameters based on the method type."""
        if params is None:
            return []
        
        params = params.copy()  # Don't modify original list
        
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
            case "getauthdataroot" | "gettotalwork" | "getfirstblockwithtotalwork":
                if len(params) > 0:
                    params[0] = str(params[0])
        
        return params

    async def send_command(self, method: str, params: list[str] = None):
        """Send a JSON-RPC command to a Zcash full node."""
        if self._session is None:
            raise RuntimeError("Session not opened. Use 'async with' or call 'await client.open()' first.")
        
        params = self._prepare_params(method, params)
        
        url = f"http://{self.rpcbind}:{self.rpcport}/"
        headers = {"Content-Type": "application/json"}
        payload = {
            "jsonrpc": "1.0",
            "id": "python-client",
            "method": method,
            "params": params
        }
        
        try:
            async with self._session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            print(f"Error communicating with the Zcash node: {e}")
            raise
    
    async def download_header(self, height: int, verbose: bool):
        """Download a block header at the given height."""
        response = await self.send_command("getblockheader", [f"{height}", (1 if verbose else 0)])
        if response["error"] is None:
            return response["result"]
        else:
            raise RuntimeError(f"Error getting block header: {response['error']}")
    
    async def download_node(self, network_upgrade: str, index: int, verbose: bool):
        """Download a history node."""
        response = await self.send_command("gethistorynode", [network_upgrade, index, (1 if verbose else 0)])
        if response["error"] is None:
            return response["result"] if verbose else bytes.fromhex(response["result"])[9:]
        else:
            raise RuntimeError(f"Error getting history node: {response['error']}")
    
    async def download_extra_data(self, command: str, height: int):
        """Download extra data using the specified command."""
        response = await self.send_command(command, [height])
        if response["error"] is None:
            return response["result"]
        else:
            raise RuntimeError(f"Error getting extra data: {response['error']}")

    async def send_commands_parallel(self, commands: list[tuple[str, list]]):
        """Send multiple commands in parallel.
        
        Args:
            commands: List of tuples (method, params)
            
        Returns:
            List of results in the same order as commands
        """
        commands = [(c, self._prepare_params(c, l)) for c, l in commands]
        url = f"http://{self.rpcbind}:{self.rpcport}/"
        headers = {"Content-Type": "application/json"}
        payloads = [{
            "jsonrpc": "1.0",
            "id": "python-client",
            "method": method,
            "params": params
        } for method, params in commands]

        async with aiohttp.ClientSession(auth=aiohttp.BasicAuth(self.rpcuser, self.rpcpassword)) as session:
            tasks = [session.post(url, headers=headers, json=payload) for payload in payloads]
            responses = await asyncio.gather(*tasks)
            json_data = await asyncio.gather(*[r.json() for r in responses])
        return json_data
    
    async def download_headers_parallel(self, block_heights: list[int], verbose: bool = True):
        """Download multiple block headers in parallel."""
        commands = [("getblockheader", [height, verbose]) for height in block_heights]
        responses = await self.send_commands_parallel(commands)
        return [r['result'] for r in responses]
        
    async def download_nodes_parallel(self, network_upgrade: str, nodes: list[int], verbose: bool):
        """Download multiple history nodes in parallel."""
        commands = [("gethistorynode", [network_upgrade, i, verbose]) for i in nodes]
        responses = await self.send_commands_parallel(commands)
        return [r['result'] for r in responses]

    async def get_first_blocks_with_total_work(self, difficulties: list[str]):
        """Get the first block with total work equal or higher than `d`, for each `d` value in `difficulties`."""
        commands = [("getfirstblockwithtotalwork", [d]) for d in difficulties]
        responses = await self.send_commands_parallel(commands)
        return [r['result'] for r in responses]

async def main():
    cmd = sys.argv[1]
    params = sys.argv[2:len(sys.argv)]
    print(params)

    # async with ZcashClient("flyclient", "", 8232, "127.0.0.1") as client:
    async with ZcashClient.from_conf(CONF_PATH) as client:
        result = await client.send_command(cmd, params)
        
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

if __name__ == "__main__":
    asyncio.run(main())
