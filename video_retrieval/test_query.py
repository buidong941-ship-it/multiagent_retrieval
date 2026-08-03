import asyncio
from config.settings import Settings
from database.milvus.milvus_client import MilvusVectorDatabase

async def main():
    settings = Settings()
    db = MilvusVectorDatabase(settings.milvus)
    # The default index does not always allow retrieving the vector. But let's test it.
    res = db.query("clip_embeddings", 'frame_id in ["L21_V001 - Trim_frame_000390"]', output_fields=["frame_id", "embedding"])
    if res:
        print("Got result, embedding shape:", len(res[0].get("embedding", [])))
    else:
        print("No result")

if __name__ == "__main__":
    asyncio.run(main())
