import sqlite3
import os
db_path = os.path.join("data", "user_docs.db")

def init_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_docs (
            user_id TEXT,
            doc_id TEXT
        )
    ''')
    conn.commit()
    conn.close()
init_db()
def user_doc(user_id: str, doc_id: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO user_docs (user_id, doc_id)
        VALUES (?, ?)
    ''', (user_id, doc_id))
    conn.commit()
    conn.close()


def get_docs_by_user(user_id: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT user_id, doc_id FROM user_docs WHERE user_id = ?
    ''', (user_id,))
    results = cursor.fetchall()
    conn.close()
    return results 

