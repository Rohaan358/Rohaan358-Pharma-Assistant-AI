import asyncio
import os
import requests
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

async def check_mongo():
    load_dotenv()
    url = os.getenv("MONGODB_URL")
    if not url:
        print("[ERROR] MONGODB_URL not found in .env")
        return False
    
    # Mask URL for safe printing
    safe_target = url.split('@')[-1] if '@' in url else "unknown"
    print(f"[INFO] Testing MongoDB connection to: {safe_target}")
    
    client = AsyncIOMotorClient(url, serverSelectionTimeoutMS=5000)
    try:
        await client.admin.command('ping')
        print("[SUCCESS] MongoDB: Connected!")
        return True
    except Exception as e:
        print(f"[ERROR] MongoDB: Connection failed: {e}")
        return False

def check_backend():
    print("[INFO] Testing Backend API (localhost:8000)...")
    try:
        response = requests.get("http://127.0.0.1:8000/", timeout=3)
        if response.status_code == 200:
            print("[SUCCESS] Backend: Reachable!")
            return True
        else:
            print(f"[ERROR] Backend: Returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Backend: Unreachable ({e})")
        return False

async def main():
    print("=== PharmaIQ Diagnostic Tool ===\n")
    db_ok = await check_mongo()
    print("-" * 30)
    api_ok = check_backend()
    print("\n" + "=" * 30)
    if db_ok and api_ok:
        print("RESULT: System looks ready for localhost!")
    else:
        print("RESULT: System has issues. Check logs above.")

if __name__ == "__main__":
    asyncio.run(main())
