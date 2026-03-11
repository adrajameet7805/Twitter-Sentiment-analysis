import sqlite3
import os
import bcrypt
import logging
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'users.db')

# ── Default admin seeds (username → plaintext password, hashed on first init) ──
_DEFAULT_ADMINS = [
    ("admin",     "admin123"),
    ("meet",      "meet123"),
    ("dataadmin", "ai2026"),
]


def get_db_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables and seed default admins if the database is empty."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                username       TEXT UNIQUE,
                email          TEXT UNIQUE,
                password_hash  TEXT NOT NULL,
                role           TEXT NOT NULL DEFAULT 'admin',
                created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login     TIMESTAMP,
                reset_token    TEXT
            )
        ''')

        # Seed default admins on first run (bcrypt)
        cursor.execute('SELECT COUNT(*) FROM users')
        if cursor.fetchone()[0] == 0:
            for uname, raw_pwd in _DEFAULT_ADMINS:
                hashed = bcrypt.hashpw(raw_pwd.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                cursor.execute(
                    'INSERT INTO users (username, email, password_hash, role, created_at) '
                    'VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)',
                    (uname, f'{uname}@dashboard.ai', hashed, 'admin')
                )
            logging.info("SQLite DB initialised with %d default admins.", len(_DEFAULT_ADMINS))

        conn.commit()
    except Exception as e:
        logging.error("Error initialising SQLite DB: %s", e)
    finally:
        conn.close()


def get_user_by_username(username: str):
    """Return the user row dict for a given username, or None if not found."""
    conn = get_db_connection()
    try:
        row = conn.execute(
            'SELECT * FROM users WHERE username = ?', (username,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_user_by_email(email: str):
    """Return the user row dict for a given email, or None if not found."""
    conn = get_db_connection()
    try:
        row = conn.execute(
            'SELECT * FROM users WHERE email = ?', (email,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_last_login(username: str):
    conn = get_db_connection()
    try:
        conn.execute(
            'UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE username = ?', (username,)
        )
        conn.commit()
    finally:
        conn.close()


def create_admin(username: str, email: str, password_hash: str):
    """Insert a new admin user into the SQLite database. Returns True on success."""
    conn = get_db_connection()
    try:
        conn.execute(
            'INSERT INTO users (username, email, password_hash, role, created_at) '
            'VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)',
            (username, email, password_hash, 'admin')
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def set_reset_token(email: str, token: str):
    conn = get_db_connection()
    try:
        conn.execute('UPDATE users SET reset_token = ? WHERE email = ?', (token, email))
        conn.commit()
    finally:
        conn.close()


def reset_password_with_token(email: str, token: str, new_password_hash: str):
    conn = get_db_connection()
    try:
        user = conn.execute(
            'SELECT reset_token FROM users WHERE email = ?', (email,)
        ).fetchone()
        if user and user['reset_token'] == token:
            conn.execute(
                'UPDATE users SET password_hash = ?, reset_token = NULL WHERE email = ?',
                (new_password_hash, email)
            )
            conn.commit()
            return True
        return False
    finally:
        conn.close()
