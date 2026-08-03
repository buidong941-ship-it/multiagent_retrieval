import asyncio
from config.settings import Settings
from database.milvus.milvus_client import MilvusVectorDatabase

async def main():
    settings = Settings()
    db = MilvusVectorDatabase(settings.milvus)
    db._connect()
    db._client.load_collection("clip_embeddings")
    res = db._client.query("clip_embeddings", filter='frame_idx > 0', output_fields=["frame_id", "embedding"], limit=1)
    if res:
        print("Got result, embedding shape:", len(res[0].get("embedding", [])))
        print("frame_id:", res[0]["frame_id"])
    else:
        print("No result")

if __name__ == "__main__":
    asyncio.run(main())
