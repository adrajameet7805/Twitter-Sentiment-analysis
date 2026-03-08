import sqlite3
import os
import bcrypt
import logging
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'users.db')

def get_db_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'admin',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                reset_token TEXT
            )
        ''')
        
        # Seed default admin if table is empty
        cursor.execute('SELECT COUNT(*) FROM users')
        if cursor.fetchone()[0] == 0:
            # Hash 'Admin@123' dynamically on first run or use predefined bcrypt salt
            hashed_pwd = bcrypt.hashpw('Admin@123'.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            cursor.execute('''
                INSERT INTO users (email, password_hash, role, created_at)
                VALUES (?, ?, 'admin', CURRENT_TIMESTAMP)
            ''', ('admin@dashboard.ai', hashed_pwd))
            logging.info("SQLite DB initialized with default admin@dashboard.ai")
            
        conn.commit()
    except Exception as e:
        logging.error(f"Error initializing SQLite DB: {e}")
    finally:
        conn.close()

def get_user_by_email(email):
    conn = get_db_connection()
    try:
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        return dict(user) if user else None
    finally:
        conn.close()

def update_last_login(email):
    conn = get_db_connection()
    try:
        conn.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE email = ?', (email,))
        conn.commit()
    finally:
        conn.close()

def create_admin(email, password_hash):
    """Insert a new admin user into the SQLite database."""
    conn = get_db_connection()
    try:
        conn.execute('''
            INSERT INTO users (email, password_hash, role, created_at)
            VALUES (?, ?, 'admin', CURRENT_TIMESTAMP)
        ''', (email, password_hash))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def set_reset_token(email, token):
    conn = get_db_connection()
    try:
        conn.execute('UPDATE users SET reset_token = ? WHERE email = ?', (token, email))
        conn.commit()
    finally:
        conn.close()

def reset_password_with_token(email, token, new_password_hash):
    conn = get_db_connection()
    try:
        user = conn.execute('SELECT reset_token FROM users WHERE email = ?', (email,)).fetchone()
        if user and user['reset_token'] == token:
            conn.execute('UPDATE users SET password_hash = ?, reset_token = NULL WHERE email = ?', 
                         (new_password_hash, email))
            conn.commit()
            return True
        return False
    finally:
        conn.close()
