import asyncio
import os
import sys

# Add project root to sys.path
sys.path.append("/Users/songtao/Projects/companion/Companion_server")

from app.db import db
from app.services.prompting.registry import PROMPT_DEFINITIONS

async def sync():
    await db.connect()
    try:
        print("Synchronizing prompts from code defaults to DB 'content'...")
        for definition in PROMPT_DEFINITIONS:
            # Update both content and defaultContent to ensure the new version is active
            await db.execute_raw(
                'UPDATE "PromptTemplate" SET "content" = $1, "defaultContent" = $1 WHERE "key" = $2',
                definition.default_text,
                definition.key
            )
            print(f"  Synced: {definition.key}")
        print("\nSuccess! All prompts have been reset to the latest code defaults.")
    except Exception as e:
        print(f"Error syncing prompts: {e}")
    finally:
        await db.disconnect()

if __name__ == "__main__":
    asyncio.run(sync())
