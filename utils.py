# === Standard Library ===
import os
import re
import json
import base64
import mimetypes
import sqlite3
import random
from pathlib import Path
from html import escape
from datetime import datetime
from urllib.parse import urljoin

# === Third-Party ===
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image  # (kept if you need it elsewhere)
from dotenv import load_dotenv
from IPython.display import HTML, display
from zai import ZhipuAiClient
import requests



# DATASET_PATH = os.path.join(os.getenv("LLM_GIT_PROJECT_LOCAL_PATH"),f"dataset{os.sep}agentic_ai")
# === Env & Clients ===
load_dotenv()

zai_api_key = os.getenv("ZHIPU_API_KEY")

# Both clients read keys from env by default; explicit is also fine:
zai_client = ZhipuAiClient(api_key=zai_api_key)

# Email API configuration
BASE_URL = os.getenv("M3_EMAIL_SERVER_API_URL",'http://localhost:5001') 
session = requests.Session()
session.headers.update({"User-Agent": "LF-ADP-EmailClient/1.0"}) 


def get_response(model: str, prompt: str, **kwargs) -> str:
    response = zai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        thinking={"type": "enabled"},
    )
    return response.choices[0].message
    
# === Data Loading ===
def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load CSV and derive date parts commonly used in charts."""
    df = pd.read_csv(csv_path)
    # Be tolerant if 'date' exists
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["quarter"] = df["date"].dt.quarter
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
    return df

# === Helpers ===
def make_schema_text(df: pd.DataFrame) -> str:
    """Return a human-readable schema from a DataFrame."""
    return "\n".join(f"- {c}: {dt}" for c, dt in df.dtypes.items())

def ensure_execute_python_tags(text: str) -> str:
    """Normalize code to be wrapped in <execute_python>...</execute_python>."""
    text = text.strip()
    # Strip ```python fences if present
    text = re.sub(r"^```(?:python)?\s*|\s*```$", "", text).strip()
    if "<execute_python>" not in text:
        text = f"<execute_python>\n{text}\n</execute_python>"
    return text

def encode_image_b64(path: str) -> tuple[str, str]:
    """Return (media_type, base64_str) for an image file path."""
    mime, _ = mimetypes.guess_type(path)
    media_type = mime or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return media_type, b64


import base64
from IPython.display import HTML, display
import pandas as pd
from typing import Any

def print_html(content: Any, title: str | None = None, is_image: bool = False):
    """
    Pretty-print inside a styled card.
    - If is_image=True and content is a string: treat as image path/URL and render <img>.
    - If content is a pandas DataFrame/Series: render as an HTML table.
    - Otherwise (strings/others): show as code/text in <pre><code>.
    """
    try:
        from html import escape as _escape
    except ImportError:
        _escape = lambda x: x

    def image_to_base64(image_path: str) -> str:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    # Render content
    if is_image and isinstance(content, str):
        b64 = image_to_base64(content)
        rendered = f'<img src="data:image/png;base64,{b64}" alt="Image" style="max-width:100%; height:auto; border-radius:8px;">'
    elif isinstance(content, pd.DataFrame):
        rendered = content.to_html(classes="pretty-table", index=False, border=0, escape=False)
    elif isinstance(content, pd.Series):
        rendered = content.to_frame().to_html(classes="pretty-table", border=0, escape=False)
    elif isinstance(content, str):
        rendered = f"<pre><code>{_escape(content)}</code></pre>"
    else:
        rendered = f"<pre><code>{_escape(str(content))}</code></pre>"

    css = """
    <style>
    .pretty-card{
      font-family: ui-sans-serif, system-ui;
      border: 2px solid transparent;
      border-radius: 14px;
      padding: 14px 16px;
      margin: 10px 0;
      background: linear-gradient(#fff, #fff) padding-box,
                  linear-gradient(135deg, #3b82f6, #9333ea) border-box;
      color: #111;
      box-shadow: 0 4px 12px rgba(0,0,0,.08);
    }
    .pretty-title{
      font-weight:700;
      margin-bottom:8px;
      font-size:14px;
      color:#111;
    }
    /* 🔒 Only affects INSIDE the card */
    .pretty-card pre, 
    .pretty-card code {
      background: #f3f4f6;
      color: #111;
      padding: 8px;
      border-radius: 8px;
      display: block;
      overflow-x: auto;
      font-size: 13px;
      white-space: pre-wrap;
    }
    .pretty-card img { max-width: 100%; height: auto; border-radius: 8px; }
    .pretty-card table.pretty-table {
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
      color: #111;
    }
    .pretty-card table.pretty-table th, 
    .pretty-card table.pretty-table td {
      border: 1px solid #e5e7eb;
      padding: 6px 8px;
      text-align: left;
    }
    .pretty-card table.pretty-table th { background: #f9fafb; font-weight: 600; }
    </style>
    """

    title_html = f'<div class="pretty-title">{title}</div>' if title else ""
    card = f'<div class="pretty-card">{title_html}{rendered}</div>'
    display(HTML(css + card))

    

    
def image_anthropic_call(model_name: str, prompt: str, media_type: str, b64: str) -> str:
    """
    Call Anthropic Claude (messages.create) with text+image and return *all* text blocks concatenated.
    Adds a system message to enforce strict JSON output.
    """
    msg = anthropic_client.messages.create(
        model=model_name,
        max_tokens=2000,
        temperature=0,
        system=(
            "You are a careful assistant. Respond with a single valid JSON object only. "
            "Do not include markdown, code fences, or commentary outside JSON."
        ),
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
            ],
        }],
    )

    # Anthropic returns a list of content blocks; collect all text
    parts = []
    for block in (msg.content or []):
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts).strip()


def image_openai_call(model_name: str, prompt: str, media_type: str, b64: str) -> str:
    data_url = f"data:{media_type};base64,{b64}"
    resp = openai_client.responses.create(
        model=model_name,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
    )
    content = (resp.output_text or "").strip()
    return content

def image_zhipu_call(model_name: str, prompt: str, media_type: str, b64: str) -> str:
    """
    Call Zhipu with text+image and return the response content.
    """
    resp = zai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "user", "content": f"<image>{b64}</image>"},
        ],
    )
    content = (resp.choices[0].message.content or "").strip()
    return content


def create_transactions_db(
    db_name: str = "products.db",
    n_products: int = 100,
    n_txns_per_product: int = 50,
) -> None:
    """
    Create an SQLite DB with a single 'transactions' table (event-sourced).
    All analytics must be derived from this table (no views).
    """
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()

    # Reset
    cur.execute("DROP TABLE IF EXISTS transactions")

    # Event-sourced transactions table
    cur.execute("""
    CREATE TABLE transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER NOT NULL,
        product_name TEXT NOT NULL,
        brand TEXT NOT NULL,
        category TEXT NOT NULL,
        color TEXT NOT NULL,

        action TEXT NOT NULL,            -- 'insert' | 'restock' | 'sale' | 'price_update'
        qty_delta INTEGER DEFAULT 0,     -- + for restock/insert, - for sale
        unit_price REAL,                 -- price at the time of the event (NULL for non-price events)
        notes TEXT,                      -- optional
        ts DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    brands = ["Nike", "Adidas", "Puma", "Reebok", "New Balance"]
    categories = ["shoes", "hoodie", "t-shirt", "hat", "backpack"]
    colors = ["black", "white", "red", "blue", "green"]

    rng = random.Random(42)
    product_catalog = []
    for pid in range(1, n_products + 1):
        name = f"{rng.choice(brands)} {rng.choice(categories)}"
        brand = name.split()[0]
        category = name.split()[1]
        color = rng.choice(colors)
        base_price = round(rng.uniform(20.0, 150.0), 2)
        product_catalog.append((pid, name, brand, category, color, base_price))

    # Seed events per product
    for (pid, name, brand, category, color, base_price) in product_catalog:
        # Initial insert (with opening stock and price)
        initial_stock = rng.randint(5, 50)
        cur.execute("""
            INSERT INTO transactions (
                product_id, product_name, brand, category, color,
                action, qty_delta, unit_price, notes
            ) VALUES (?, ?, ?, ?, ?, 'insert', ?, ?, ?)
        """, (pid, name, brand, category, color, initial_stock, base_price,
              f"Initial insert with stock={initial_stock}, price={base_price}"))

        current_price = base_price

        # Follow-up events
        for _ in range(n_txns_per_product - 1):
            event_type = rng.choices(
                ["restock", "sale", "price_update"],
                weights=[0.25, 0.6, 0.15],
                k=1
            )[0]

            if event_type == "restock":
                qty = rng.randint(1, 25)
                cur.execute("""
                    INSERT INTO transactions (
                        product_id, product_name, brand, category, color,
                        action, qty_delta, unit_price, notes
                    ) VALUES (?, ?, ?, ?, ?, 'restock', ?, NULL, ?)
                """, (pid, name, brand, category, color, qty,
                      f"Restock +{qty} units"))

            elif event_type == "sale":
                qty = -rng.randint(1, 10)  # negative
                cur.execute("""
                    INSERT INTO transactions (
                        product_id, product_name, brand, category, color,
                        action, qty_delta, unit_price, notes
                    ) VALUES (?, ?, ?, ?, ?, 'sale', ?, ?, ?)
                """, (pid, name, brand, category, color, qty, current_price,
                      f"Sale {-qty} units at {current_price}"))

            else:  # price_update
                delta = round(rng.uniform(-5.0, 5.0), 2)
                current_price = max(1.0, round(current_price + delta, 2))
                cur.execute("""
                    INSERT INTO transactions (
                        product_id, product_name, brand, category, color,
                        action, qty_delta, unit_price, notes
                    ) VALUES (?, ?, ?, ?, ?, 'price_update', 0, ?, ?)
                """, (pid, name, brand, category, color, current_price,
                      f"Price update to {current_price}"))

    conn.commit()
    conn.close()

    print(f"SQLite database '{db_name}' created with a single 'transactions' table (event-sourced).")


def get_schema(db_path: str) -> str:
    """
    Return only the schema that the agent should use: 'transactions' table.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(transactions)")
    rows = cur.fetchall()
    conn.close()
    return "table name: transactions\n" + "\n".join([f"{r[1]} ({r[2]})" for r in rows])

# ================================
# Email API Functions
# ================================
def pretty_display(title: str, response: requests.Response):
    """Render an HTTP response in a styled block; returns parsed content (JSON if possible)."""
    status = response.status_code
    try:
        content = response.json()
        body = json.dumps(content, indent=2)
    except Exception:
        content = response.text
        body = content

    html = f"""
    <div style='border:1px solid #ccc; border-left:5px solid #007bff; padding:10px; margin:10px 0; background:#f9f9f9; color:#000;'>
        <strong style='color:#007bff'>{escape(title)}:</strong>
        <span style='color:{"green" if status == 200 else "red"}'> Status {status}</span>
        <pre style='font-size:12px; margin-top:10px; white-space:pre-wrap; color:#000;'>{escape(body)}</pre>
    </div>
    """
    display(HTML(html))
    return content

def reset_database() -> dict:
    """Calls /reset_database endpoint and returns confirmation message."""
    r = session.get(f"{BASE_URL}/reset_database")
    r.raise_for_status()
    return r.json()

def test_send_email():
    payload = {
        "recipient": "test@example.com",
        "subject": "Test Subject",
        "body": "This is a test email body.",
    }
    r = session.post(f"{BASE_URL}/send", json=payload)
    return pretty_display("POST /send", r)

def test_list_emails():
    r = session.get(f"{BASE_URL}/emails")
    return pretty_display("GET /emails", r)

def test_search_emails(q: str = "report"):
    r = session.get(f"{BASE_URL}/emails/search", params={"q": q})
    return pretty_display(f"GET /emails/search?q={q}", r)

def test_filter_emails(recipient: str | None = None, date_from: str | None = None, date_to: str | None = None):
    params = {}
    if recipient:
        params["recipient"] = recipient
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    r = session.get(f"{BASE_URL}/emails/filter", params=params)
    return pretty_display("GET /emails/filter", r)

def test_unread_emails():
    r = session.get(f"{BASE_URL}/emails/unread")
    return pretty_display("GET /emails/unread", r)

def test_get_email(email_id: str):
    r = session.get(f"{BASE_URL}/emails/{email_id}")
    return pretty_display(f"GET /emails/{email_id}", r)

def test_mark_read(email_id: str):
    r = session.patch(f"{BASE_URL}/emails/{email_id}/read")
    return pretty_display(f"PATCH /emails/{email_id}/read", r)

def test_mark_unread(email_id: str):
    r = session.patch(f"{BASE_URL}/emails/{email_id}/unread")
    return pretty_display(f"PATCH /emails/{email_id}/unread", r)

def test_delete_email(email_id: str):
    r = session.delete(f"{BASE_URL}/emails/{email_id}")
    return pretty_display(f"DELETE /emails/{email_id}", r)

def call_llm_email_agent(prompt: str,
                         api_url: str | None = None,
                         timeout: int = 30) -> dict:
    """
    Calls the M3 LLM server with a natural-language instruction.

    Args:
        prompt: Instruction for the agent (e.g., "Check unread emails...").
        api_url: Base URL of the LLM server. If None, uses env var M3_LLM_SERVER_URL.
        timeout: HTTP timeout in seconds.

    Returns:
        dict with keys: ok (bool), status (int), response (str|None), raw (dict|str)
    """
    # Resolve API base URL
    base = api_url or os.getenv("M3_LLM_SERVER_URL")
    if not base:
        raise RuntimeError("M3_LLM_SERVER_URL is not set. Put it in your .env (e.g., http://127.0.0.1:5001).")

    # Build final endpoint; accept both with/without trailing /prompt
    endpoint = base if base.rstrip("/").endswith("/prompt") else urljoin(base.rstrip("/") + "/", "prompt")

    try:
        r = requests.post(endpoint, json={"prompt": prompt}, timeout=timeout)
    except requests.RequestException as e:
        return {"ok": False, "status": None, "response": None, "raw": str(e)}

    try:
        data = r.json()
    except ValueError:
        data = r.text

    ok = (r.status_code == 200)
    return {"ok": ok, "status": r.status_code, "response": (data.get("response") if isinstance(data, dict) else None), "raw": data}


def execute_sql(query: str, db_path: str) -> pd.DataFrame:
    """
    Execute any SELECT over the event-sourced 'transactions' table.
    """
    q = query.strip().removeprefix("```sql").removesuffix("```").strip()
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(q, conn)
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})
    finally:
        conn.close()
