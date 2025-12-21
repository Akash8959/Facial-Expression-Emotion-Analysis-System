import sqlite3
import os

db_path = "facial_emotion.db"

if not os.path.exists(db_path):
    print("❌ Database not found! Run your Flask app at least once to create it.")
else:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("\n--- 👥 Registered Users ---")
    for row in cursor.execute("SELECT id, name FROM users"):
        print(f"ID: {row[0]}, Username: {row[1]}")


    conn.close()
