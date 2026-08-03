import asyncio
from pipelines.online_pipeline import OnlinePipeline

async def test():
    p = OnlinePipeline()
    await p.initialize()
    res = await p.retrieve("biển báo", mode="clip")
    print("Found:", len(res))

if __name__ == "__main__":
    asyncio.run(test())
