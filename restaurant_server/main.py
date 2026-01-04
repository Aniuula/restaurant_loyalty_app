from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(APP_DIR, "loyalty.sqlite3")

# Ustaw pod Twój model na telefonie
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "192"))

# cosine distance = 1 - cosine_similarity; im mniejszy tym lepiej
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.35"))

# Po ilu wizytach nagroda
REWARD_EVERY = int(os.getenv("REWARD_EVERY", "5"))

app = FastAPI(title="Restaurant Loyalty (Face Embeddings)", version="1.1.0")


# ---------- DB ----------
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    conn = get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS customers (
            id TEXT PRIMARY KEY,
            display_name TEXT,
            embedding BLOB NOT NULL,
            visits_total INTEGER NOT NULL DEFAULT 0,
            visits_since_reward INTEGER NOT NULL DEFAULT 0,
            last_visit_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def ensure_schema() -> None:
    """Bezpieczna migracja: dodaje brakujące kolumny, nie kasuje danych."""
    conn = get_conn()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(customers)").fetchall()]
    if "last_visit_at" not in cols:
        conn.execute("ALTER TABLE customers ADD COLUMN last_visit_at TEXT")
    conn.commit()
    conn.close()


init_db()
ensure_schema()


# ---------- Models ----------
class EmbeddingPayload(BaseModel):
    embedding: List[float] = Field(..., description="Embedding wektora twarzy z telefonu")
    device_id: Optional[str] = None


class EnrollPayload(BaseModel):
    # Możesz przesyłać 1 embedding lub kilka (np. 3-5) i backend uśredni
    embeddings: List[List[float]] = Field(..., min_items=1)
    display_name: Optional[str] = None


class ScanResponse(BaseModel):
    status: str  # matched | not_found
    state_label: Optional[str] = None  # "nowy klient" | "stały klient"
    customer_id: Optional[str] = None
    display_name: Optional[str] = None
    visits_total: Optional[int] = None
    visits_since_reward: Optional[int] = None
    reward: bool = False
    reward_name: Optional[str] = None
    match_distance: Optional[float] = None
    last_visit_at: Optional[str] = None  # ISO UTC


class EnrollResponse(BaseModel):
    status: str  # created
    customer_id: str
    display_name: Optional[str]
    visits_total: int
    visits_since_reward: int
    last_visit_at: Optional[str]


# ---------- Vector utils ----------
def to_vec(emb: List[float]) -> np.ndarray:
    v = np.asarray(emb, dtype=np.float32)
    if v.ndim != 1 or v.shape[0] != EMBEDDING_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"Zły rozmiar embeddingu. Oczekiwano {EMBEDDING_DIM}, dostałem {v.shape}.",
        )
    norm = np.linalg.norm(v)
    if norm == 0:
        raise HTTPException(status_code=400, detail="Embedding ma normę 0.")
    return v / norm


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    sim = float(np.dot(a, b))
    return 1.0 - sim


def blob_from_vec(v: np.ndarray) -> bytes:
    return v.astype(np.float32).tobytes()


def vec_from_blob(blob: bytes) -> np.ndarray:
    v = np.frombuffer(blob, dtype=np.float32)
    if v.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"Embedding w DB ma zły wymiar: {v.shape[0]} != {EMBEDDING_DIM}")
    return v


def find_best_match(conn: sqlite3.Connection, query_vec: np.ndarray) -> Tuple[Optional[dict], Optional[float]]:
    rows = conn.execute(
        """
        SELECT id, display_name, embedding, visits_total, visits_since_reward, last_visit_at
        FROM customers
        """
    ).fetchall()

    if not rows:
        return None, None

    best = None
    best_dist = 999.0

    for (cid, name, emb_blob, vt, vsr, last_visit_at) in rows:
        db_vec = vec_from_blob(emb_blob)
        dist = cosine_distance(query_vec, db_vec)
        if dist < best_dist:
            best_dist = dist
            best = {
                "id": cid,
                "display_name": name,
                "visits_total": int(vt),
                "visits_since_reward": int(vsr),
                "last_visit_at": last_visit_at,
            }

    return best, float(best_dist)


# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {
        "ok": True,
        "embedding_dim": EMBEDDING_DIM,
        "match_threshold": MATCH_THRESHOLD,
        "reward_every": REWARD_EVERY,
    }


@app.post("/enroll", response_model=EnrollResponse)
def enroll(payload: EnrollPayload):
    vecs = [to_vec(e) for e in payload.embeddings]
    mean_vec = np.mean(np.stack(vecs, axis=0), axis=0)
    mean_vec = mean_vec / np.linalg.norm(mean_vec)

    cid = str(uuid.uuid4())
    ts = now_iso()

    conn = get_conn()
    conn.execute(
        """
        INSERT INTO customers (
            id, display_name, embedding,
            visits_total, visits_since_reward,
            last_visit_at, created_at, updated_at
        )
        VALUES (?, ?, ?, 0, 0, NULL, ?, ?)
        """,
        (cid, payload.display_name, blob_from_vec(mean_vec), ts, ts),
    )
    conn.commit()
    conn.close()

    return EnrollResponse(
        status="created",
        customer_id=cid,
        display_name=payload.display_name,
        visits_total=0,
        visits_since_reward=0,
        last_visit_at=None,
    )


@app.post("/scan-visit", response_model=ScanResponse)
def scan_visit(payload: EmbeddingPayload):
    q = to_vec(payload.embedding)

    conn = get_conn()
    best, dist = find_best_match(conn, q)

    if best is None or dist is None or dist > MATCH_THRESHOLD:
        conn.close()
        return ScanResponse(status="not_found", match_distance=dist)

    # nalicz wizytę
    new_total = best["visits_total"] + 1
    new_since = best["visits_since_reward"] + 1

    reward = False
    reward_name = None
    if new_since >= REWARD_EVERY:
        reward = True
        reward_name = "Darmowa kawa"
        new_since = 0

    visit_time = now_iso()

    conn.execute(
        """
        UPDATE customers
        SET visits_total = ?,
            visits_since_reward = ?,
            last_visit_at = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (new_total, new_since, visit_time, now_iso(), best["id"]),
    )
    conn.commit()
    conn.close()

    state_label = "nowy klient" if new_total == 1 else "stały klient"

    return ScanResponse(
        status="matched",
        state_label=state_label,
        customer_id=best["id"],
        display_name=best["display_name"],
        visits_total=new_total,
        visits_since_reward=new_since,
        reward=reward,
        reward_name=reward_name,
        match_distance=dist,
        last_visit_at=visit_time,
    )


@app.get("/customers")
def list_customers():
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT id, display_name, visits_total, visits_since_reward, last_visit_at, created_at, updated_at
        FROM customers
        ORDER BY updated_at DESC
        """
    ).fetchall()
    conn.close()

    return [
        {
            "id": r[0],
            "display_name": r[1],
            "visits_total": r[2],
            "visits_since_reward": r[3],
            "last_visit_at": r[4],
            "created_at": r[5],
            "updated_at": r[6],
        }
        for r in rows
    ]


@app.delete("/customers/{customer_id}")
def delete_customer(customer_id: str):
    conn = get_conn()
    cur = conn.execute("DELETE FROM customers WHERE id = ?", (customer_id,))
    conn.commit()
    conn.close()
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Nie ma takiego klienta.")
    return {"status": "deleted", "customer_id": customer_id}
