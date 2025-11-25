from demo import FlyclientDemo
from zcash_client import ZcashClient, CONF_PATH
import asyncio

async def main():
    # async with ZcashClient("flyclient", "", 8232, "127.0.0.1") as client:
    async with ZcashClient.from_conf(CONF_PATH) as client:
        proof = FlyclientDemo.from_file(client, 'proof.json')
        if proof.verify_non_interactive():
            print('Success!')
        else:
            print("Failure")

if __name__ == "__main__":
    asyncio.run(main())