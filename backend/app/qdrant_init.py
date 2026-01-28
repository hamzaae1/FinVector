import os
import time
import requests

QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION = os.getenv("QDRANT_COLLECTION")
SNAPSHOT_PATH = os.getenv("SNAPSHOT_PATH")

def wait_for_qdrant():
    for _ in range(20):
        try:
            requests.get(QDRANT_URL)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("Qdrant not available")


def restore_snapshot():
    r = requests.get(f"{QDRANT_URL}/collections/{COLLECTION}")
    if r.status_code == 200:
        print("✅ Collection already exists")
        return

    print("🔄 Restoring snapshot...")
    requests.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/snapshots/recover",
        json={"location": SNAPSHOT_PATH},
        timeout=60,
    )