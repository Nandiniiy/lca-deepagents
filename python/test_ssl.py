import truststore
truststore.inject_into_ssl()

import httpx
import asyncio

async def test():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://docs.langchain.com")
        print(response.status_code)

asyncio.run(test())