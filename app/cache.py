import time
from typing import Optional

import psycopg
from openai import AsyncOpenAI


class ResponseCache:
    """
    Semantic caching using Neon PostgreSQL, pgvector, and OpenAI Embeddings.
    """

    def __init__(
        self,
        db_url: str,
        openai_api_key: str,
        ttl_seconds: int = 300,
        similarity_threshold: float = 0.15,
    ):
        self.db_url = db_url
        self.ttl = ttl_seconds
        self.threshold = similarity_threshold
        # Initialize the OpenAI client for generating embeddings
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self._hits = 0
        self._misses = 0

    async def setup(self) -> None:
        """Create the semantic cache table and ensure pgvector is active."""
        async with await psycopg.AsyncConnection.connect(self.db_url) as conn:
            async with conn.cursor() as cur:
                # Double-check pgvector is enabled on this Neon database
                await cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

                # Create the semantic table (Notice the VECTOR(1536) column)
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS semantic_cache (
                        id SERIAL PRIMARY KEY,
                        query TEXT NOT NULL,
                        response TEXT NOT NULL,
                        embedding VECTOR(1536),
                        timestamp FLOAT NOT NULL
                    )
                """)
                await conn.commit()

    async def _get_embedding(self, text: str) -> list[float]:
        """Call OpenAI to generate a 1,536-dimension vector array for the text."""
        response = await self.client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return response.data[0].embedding

    async def get(self, query: str) -> Optional[str]:
        """Semantic search using Cosine Distance."""
        # 1. Turn the user's string query into math
        query_vector = await self._get_embedding(query)

        async with await psycopg.AsyncConnection.connect(self.db_url) as conn:
            async with conn.cursor() as cur:
                # 2. Find the closest match in Neon using <=> (Cosine Distance)
                # We cast the python list to a vector using ::vector
                await cur.execute(
                    """
                    SELECT response, timestamp, (embedding <=> %s::vector) AS distance
                    FROM semantic_cache
                    ORDER BY distance ASC
                    LIMIT 1
                """,
                    (query_vector,),
                )

                row = await cur.fetchone()

                if row:
                    response_text, timestamp, distance = row

                    # 3. Check if it means the same thing AND hasn't expired
                    if distance <= self.threshold and (
                        time.time() - timestamp < self.ttl
                    ):
                        self._hits += 1
                        return response_text

        self._misses += 1
        return None

    async def set(self, query: str, response: str) -> None:
        """Cache the response alongside its semantic embedding."""
        embedding = await self._get_embedding(query)
        timestamp = time.time()

        async with await psycopg.AsyncConnection.connect(self.db_url) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO semantic_cache (query, response, embedding, timestamp)
                    VALUES (%s, %s, %s::vector, %s)
                """,
                    (query, response, embedding, timestamp),
                )
                await conn.commit()
