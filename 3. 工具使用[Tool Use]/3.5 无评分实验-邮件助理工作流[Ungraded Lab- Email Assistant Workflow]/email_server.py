from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
from datetime import datetime

app = FastAPI(title="M3 Email Server")
DB_PATH = "emails.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

class SendEmailRequest(BaseModel):
    recipient: str
    subject: str
    body: str

@app.post("/send")
def send_email(req: SendEmailRequest):
    conn = get_db()
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute('''
        INSERT INTO emails (sender, recipient, subject, body, timestamp, read)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', ('you@mail.com', req.recipient, req.subject, req.body, timestamp, False))
    email_id = c.lastrowid
    conn.commit()
    c.execute("SELECT * FROM emails WHERE id = ?", (email_id,))
    row = dict(c.fetchone())
    conn.close()
    return row

@app.get("/emails")
def list_emails():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM emails ORDER BY timestamp DESC")
    emails = [dict(row) for row in c.fetchall()]
    conn.close()
    return emails

@app.get("/emails/unread")
def list_unread():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM emails WHERE read = 0 ORDER BY timestamp DESC")
    emails = [dict(row) for row in c.fetchall()]
    conn.close()
    return emails

@app.get("/emails/search")
def search_emails(q: str):
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        SELECT * FROM emails
        WHERE subject LIKE ? OR body LIKE ? OR sender LIKE ?
        ORDER BY timestamp DESC
    ''', (f'%{q}%', f'%{q}%', f'%{q}%'))
    emails = [dict(row) for row in c.fetchall()]
    conn.close()
    return emails

@app.get("/emails/{email_id}")
def get_email(email_id: int):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM emails WHERE id = ?", (email_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Email not found")
    return dict(row)

@app.patch("/emails/{email_id}/read")
def mark_read(email_id: int):
    conn = get_db()
    c = conn.cursor()
    c.execute("UPDATE emails SET read = 1 WHERE id = ?", (email_id,))
    conn.commit()
    c.execute("SELECT * FROM emails WHERE id = ?", (email_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Email not found")
    return dict(row)

@app.patch("/emails/{email_id}/unread")
def mark_unread(email_id: int):
    conn = get_db()
    c = conn.cursor()
    c.execute("UPDATE emails SET read = 0 WHERE id = ?", (email_id,))
    conn.commit()
    c.execute("SELECT * FROM emails WHERE id = ?", (email_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Email not found")
    return dict(row)

@app.delete("/emails/{email_id}")
def delete_email(email_id: int):
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM emails WHERE id = ?", (email_id,))
    conn.commit()
    conn.close()
    return {"message": "Email deleted"}

@app.get("/reset_database")
def reset_database():
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM emails")
    conn.commit()
    conn.close()
    return {"message": "Database reset"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=5001)
