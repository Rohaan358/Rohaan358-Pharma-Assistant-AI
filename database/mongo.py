"""
database/mongo.py — Async MongoDB connection and CRUD helpers using Motor
"""
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING
from typing import Optional, List, Dict, Any
import logging
from config import get_settings

logger = logging.getLogger(__name__)

# Global client reference
_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None




async def connect_to_mongodb(db_name: str = "pharma_db") -> bool:
    """Connect to MongoDB using URL from settings."""
    global _client, _db
    try:
        settings = get_settings()
        url = settings.mongodb_url
        if not url:
            logger.error("MONGODB_URL not found in settings")
            return False
            
        url = url.strip()
        # Masked URL for logs
        host_part = url.split("@")[-1] if "@" in url else url
        logger.info(f"Connecting to MongoDB Cluster: {host_part}")
        
        _client = AsyncIOMotorClient(
            url, 
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000
        )
        
        # Verify connection
        await _client.admin.command("ping")
        
        _db = _client[db_name]
        logger.info(f"MongoDB connected successfully to database: '{db_name}'")
        return True
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        _client = None
        _db = None
        return False


async def close_mongo_connection():
    """Close the MongoDB connection."""
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
        logger.info("MongoDB connection closed.")


def get_database() -> AsyncIOMotorDatabase:
    """Return the active database instance."""
    if _db is None:
        raise RuntimeError("Not connected to MongoDB. Call /connect first.")
    return _db


def is_connected() -> bool:
    return _db is not None


# ─── CRUD Helpers ────────────────────────────────────────────────────────────

async def insert_many_documents(collection: str, documents: List[Dict]) -> int:
    db = get_database()
    result = await db[collection].insert_many(documents)
    return len(result.inserted_ids)


async def find_documents(
    collection: str,
    query: Dict = {},
    projection: Optional[Dict] = None,
    sort_field: Optional[str] = None,
    sort_order: int = ASCENDING,
    limit: int = 0,
) -> List[Dict]:
    db = get_database()
    cursor = db[collection].find(query, projection)
    if sort_field:
        cursor = cursor.sort(sort_field, sort_order)
    if limit:
        cursor = cursor.limit(limit)
    return await cursor.to_list(length=None)


async def aggregate_documents(collection: str, pipeline: List[Dict]) -> List[Dict]:
    db = get_database()
    cursor = db[collection].aggregate(pipeline)
    return await cursor.to_list(length=None)


async def count_documents(collection: str, query: Dict = {}) -> int:
    db = get_database()
    return await db[collection].count_documents(query)


async def upsert_document(collection: str, filter_query: Dict, update_data: Dict) -> bool:
    db = get_database()
    result = await db[collection].update_one(
        filter_query, {"$set": update_data}, upsert=True
    )
    return result.acknowledged


async def delete_documents(collection: str, query: Dict) -> int:
    db = get_database()
    result = await db[collection].delete_many(query)
    return result.deleted_count


async def get_distinct_values(collection: str, field: str, query: Dict = {}) -> List:
    db = get_database()
    return await db[collection].distinct(field, query)


# ─── Chat History Helpers ────────────────────────────────────────────────────

async def save_chat(document: Dict[str, Any]) -> bool:
    """Save a single chat message to chat_history collection."""
    db = get_database()
    result = await db["chat_history"].insert_one(document)
    return result.acknowledged


async def get_chat_history(session_id: str) -> List[Dict[str, Any]]:
    """Return all messages for a session sorted by timestamp ASC."""
    return await find_documents(
        "chat_history",
        {"session_id": session_id},
        projection={"_id": 0},
        sort_field="timestamp",
        sort_order=ASCENDING
    )


async def get_all_sessions() -> List[Dict[str, Any]]:
    """Return summary of all stored sessions."""
    pipeline = [
        {
            "$group": {
                "_id": "$session_id",
                "start_time": {"$min": "$timestamp"},
                "message_count": {"$sum": 1},
                "first_message": {"$first": "$message"},
                "context": {"$first": "$context"}
            }
        },
        {
            "$project": {
                "_id": 0,
                "session_id": "$_id",
                "start_time": 1,
                "message_count": 1,
                "first_message": 1,
                "context": 1
            }
        },
        {"$sort": {"start_time": DESCENDING}}
    ]
    return await aggregate_documents("chat_history", pipeline)


async def delete_session(session_id: str) -> int:
    """Delete all documents associated with a session_id."""
    return await delete_documents("chat_history", {"session_id": session_id})
