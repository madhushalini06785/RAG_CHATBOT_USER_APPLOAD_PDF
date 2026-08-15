import sqlite3
import os
from datetime import datetime


# ==================================================
# DATABASE PATH
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(
    BASE_DIR,
    "chat_history.db"
)


# ==================================================
# DATABASE CONNECTION
# ==================================================

def get_connection():

    conn = sqlite3.connect(
        DB_PATH,
        check_same_thread=False
    )

    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON")

    return conn


# ==================================================
# INITIALIZE DATABASE
# ==================================================

def init_database():

    conn = get_connection()
    cursor = conn.cursor()

    # --------------------------------------------------
    # CHATS
    # --------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (

            chat_id TEXT PRIMARY KEY,

            title TEXT NOT NULL,

            namespace TEXT NOT NULL,

            created_at TEXT NOT NULL,

            updated_at TEXT NOT NULL
        )
    """)

    # --------------------------------------------------
    # MESSAGES
    # --------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            chat_id TEXT NOT NULL,

            role TEXT NOT NULL,

            content TEXT NOT NULL,

            created_at TEXT NOT NULL,

            FOREIGN KEY(chat_id)
                REFERENCES chats(chat_id)
                ON DELETE CASCADE
        )
    """)

    # --------------------------------------------------
    # DOCUMENTS
    # --------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            chat_id TEXT NOT NULL,

            filename TEXT NOT NULL,

            file_size INTEGER,

            pages INTEGER,

            chunks INTEGER,

            uploaded_at TEXT NOT NULL,

            FOREIGN KEY(chat_id)
                REFERENCES chats(chat_id)
                ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()


# ==================================================
# CREATE CHAT
# ==================================================

def create_chat(
    chat_id,
    title,
    namespace
):

    conn = get_connection()
    cursor = conn.cursor()

    now = datetime.now().isoformat()

    cursor.execute(
        """
        INSERT INTO chats
        (
            chat_id,
            title,
            namespace,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            chat_id,
            title,
            namespace,
            now,
            now
        )
    )

    conn.commit()
    conn.close()


# ==================================================
# GET ALL CHATS
# ==================================================

def get_all_chats():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            chat_id,
            title,
            namespace,
            created_at,
            updated_at

        FROM chats

        ORDER BY updated_at DESC
        """
    )

    rows = cursor.fetchall()

    conn.close()

    chats = []

    for row in rows:

        chats.append({

            "chat_id": row[0],

            "title": row[1],

            "namespace": row[2],

            "created_at": row[3],

            "updated_at": row[4]
        })

    return chats


# ==================================================
# GET SINGLE CHAT
# ==================================================

def get_chat(chat_id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            chat_id,
            title,
            namespace,
            created_at,
            updated_at

        FROM chats

        WHERE chat_id = ?
        """,
        (chat_id,)
    )

    row = cursor.fetchone()

    conn.close()

    if row is None:
        return None

    return {

        "chat_id": row[0],

        "title": row[1],

        "namespace": row[2],

        "created_at": row[3],

        "updated_at": row[4]
    }


# ==================================================
# UPDATE CHAT TITLE
# ==================================================

def update_chat_title(
    chat_id,
    title
):

    conn = get_connection()
    cursor = conn.cursor()

    now = datetime.now().isoformat()

    cursor.execute(
        """
        UPDATE chats

        SET
            title = ?,
            updated_at = ?

        WHERE chat_id = ?
        """,
        (
            title,
            now,
            chat_id
        )
    )

    conn.commit()
    conn.close()


# ==================================================
# UPDATE CHAT TIMESTAMP
# ==================================================

def update_chat_timestamp(chat_id):

    conn = get_connection()
    cursor = conn.cursor()

    now = datetime.now().isoformat()

    cursor.execute(
        """
        UPDATE chats

        SET updated_at = ?

        WHERE chat_id = ?
        """,
        (
            now,
            chat_id
        )
    )

    conn.commit()
    conn.close()


# ==================================================
# SAVE MESSAGE
# ==================================================

def save_message(
    chat_id,
    role,
    content
):

    conn = get_connection()
    cursor = conn.cursor()

    now = datetime.now().isoformat()

    cursor.execute(
        """
        INSERT INTO messages
        (
            chat_id,
            role,
            content,
            created_at
        )

        VALUES (?, ?, ?, ?)
        """,
        (
            chat_id,
            role,
            content,
            now
        )
    )

    cursor.execute(
        """
        UPDATE chats

        SET updated_at = ?

        WHERE chat_id = ?
        """,
        (
            now,
            chat_id
        )
    )

    conn.commit()
    conn.close()


# ==================================================
# GET CHAT MESSAGES
# ==================================================

def get_messages(chat_id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            role,
            content

        FROM messages

        WHERE chat_id = ?

        ORDER BY id ASC
        """,
        (chat_id,)
    )

    rows = cursor.fetchall()

    conn.close()

    return [

        {
            "role": row[0],
            "content": row[1]
        }

        for row in rows
    ]


# ==================================================
# SAVE DOCUMENT
# ==================================================

def save_document(
    chat_id,
    filename,
    file_size,
    pages,
    chunks
):

    conn = get_connection()
    cursor = conn.cursor()

    now = datetime.now().isoformat()

    cursor.execute(
        """
        INSERT INTO documents
        (
            chat_id,
            filename,
            file_size,
            pages,
            chunks,
            uploaded_at
        )

        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            chat_id,
            filename,
            file_size,
            pages,
            chunks,
            now
        )
    )

    conn.commit()
    conn.close()


# ==================================================
# GET DOCUMENTS
# ==================================================

def get_documents(chat_id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            filename,
            file_size,
            pages,
            chunks,
            uploaded_at

        FROM documents

        WHERE chat_id = ?

        ORDER BY id ASC
        """,
        (chat_id,)
    )

    rows = cursor.fetchall()

    conn.close()

    return [

        {
            "name": row[0],
            "size": row[1],
            "pages": row[2],
            "chunks": row[3],
            "uploaded_at": row[4]
        }

        for row in rows
    ]


# ==================================================
# DELETE CHAT
# ==================================================

def delete_chat(chat_id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM chats

        WHERE chat_id = ?
        """,
        (chat_id,)
    )

    conn.commit()
    conn.close()


# ==================================================
# CLEAR MESSAGES
# ==================================================

def clear_messages(chat_id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM messages

        WHERE chat_id = ?
        """,
        (chat_id,)
    )

    conn.commit()
    conn.close()


# ==================================================
# INITIALIZE
# ==================================================

init_database()
