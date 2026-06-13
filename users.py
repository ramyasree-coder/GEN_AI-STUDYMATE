import os
import sqlite3
import hashlib
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")


def _get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL
    )
    """
    )
    conn.commit()
    conn.close()


def _hash_password(password: str, salt: str = "study_salt") -> str:
    h = hashlib.sha256()
    h.update((salt + password).encode("utf-8"))
    return h.hexdigest()


def create_user(username: str, password: str) -> bool:
    conn = _get_conn()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, _hash_password(password)))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def verify_user(username: str, password: str) -> bool:
    conn = _get_conn()
    cur = conn.cursor()
    try:
        cur.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        if not row:
            return False
        return _hash_password(password) == row[0]
    finally:
        conn.close()


def user_exists(username: str) -> bool:
    conn = _get_conn()
    cur = conn.cursor()
    try:
        cur.execute("SELECT 1 FROM users WHERE username = ?", (username,))
        return cur.fetchone() is not None
    finally:
        conn.close()
