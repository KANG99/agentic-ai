# === Standard Library ===
import os
import re
import json
import base64
import mimetypes
import sqlite3
import random
import urllib.parse
from pathlib import Path
from html import escape
from datetime import datetime
from typing import Any
from urllib.parse import urljoin

# === Third-Party ===
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image  # (kept if you need it elsewhere)
from dotenv import load_dotenv
from IPython.display import HTML, display
from zai import ZhipuAiClient
from openai import OpenAI
import requests



# DATASET_PATH = os.path.join(os.getenv("LLM_GIT_PROJECT_LOCAL_PATH"),f"dataset{os.sep}agentic_ai")
# === Env & Clients ===
load_dotenv()

zai_api_key = os.getenv("ZHIPU_API_KEY")

# Both clients read keys from env by default; explicit is also fine:
zai_client = ZhipuAiClient(api_key=zai_api_key)

qwen_client_ollama = OpenAI(base_url="http://localhost:11434/v1",api_key="sk-no-key-required")
qwen_client_llama = OpenAI(base_url="http://127.0.0.1:8080",api_key="sk-no-key-required")


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


# ================================
# Component-level Evaluation Functions
# ================================

_URL_RE = re.compile(r"https?://[^\s\)\]\}<>\"']+", re.IGNORECASE)


def clean_json_block(raw: str) -> str:
    """Remove markdown code fences from a JSON string."""
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


def _extract_hostname(url: str) -> str:
    """Extract hostname from a URL, stripping 'www.' prefix if present."""
    try:
        host = urllib.parse.urlparse(url).hostname or ""
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def extract_urls(text: str) -> list[dict[str, Any]]:
    """
    Best-effort URL extractor from arbitrary text.
    Returns list of {title, url, source} dicts (title/source may be None).
    """
    if not isinstance(text, str):
        text = str(text)
    urls = _URL_RE.findall(text)
    items = []
    for u in urls:
        host = _extract_hostname(u)
        items.append({"title": None, "url": u, "source": host or None})
    return items


def evaluate_anytext_against_domains(TOP_DOMAINS: set[str], payload: Any, min_ratio: float = 0.4):
    """
    Accepts:
      - raw list[dict] (Tavily-like), or
      - raw string (free text with links), or
      - dict with 'results' list
    Returns (ok, report_dict), same shape as before.
    """
    # Normalize into items: list[dict(title,url,source)]
    items = []
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict) and isinstance(payload.get("results"), list):
        items = payload["results"]
    elif isinstance(payload, str):
        # try JSON first
        s = payload.strip()
        if s.startswith("```"):
            s = re.sub(r"^```(?:json|text|markdown)?\s*", "", s)
            s = re.sub(r"\s*```$", "", s)
        try:
            maybe = json.loads(s)
            if isinstance(maybe, list):
                items = maybe
            else:
                items = extract_urls(payload)
        except Exception:
            items = extract_urls(payload)
    else:
        items = extract_urls(str(payload))

    total = len(items)
    if total == 0:
        return False, {"total": 0, "approved": 0, "ratio": 0.0, "details": [], "note": "No items/links parsed"}

    details = []
    approved = 0
    for it in items:
        url = (it or {}).get("url")
        host = _extract_hostname(url or "")
        ok = any(host.endswith(dom) for dom in TOP_DOMAINS) if host else False
        if ok:
            approved += 1
        details.append({
            "title": (it or {}).get("title"),
            "url": url,
            "host": host,
            "approved": ok,
        })

    ratio = approved / max(total, 1)
    ok = ratio >= min_ratio
    return ok, {"total": total, "approved": approved, "ratio": ratio, "details": details, "min_ratio": min_ratio}


def evaluate_references(history: list[tuple[str, str, str]], TOP_DOMAINS: set[str], min_ratio: float = 0.4) -> str:
    """
    Pure evaluator. Finds the most recent research_agent output (any text or JSON),
    extracts links, compares domains to TOP_DOMAINS, and returns a Markdown PASS/FAIL.
    """
    # 1) Prefer latest research_agent output
    payload = None
    for step, agent, output in reversed(history):
        if agent == "research_agent":
            payload = output
            break
    # 2) Fallback: any output with links or array-looking text
    if payload is None:
        for _, _, output in reversed(history):
            if isinstance(output, str) and (("http://" in output) or ("https://" in output) or ("[" in output and "]" in output)):
                payload = output
                break

    if payload is None:
        ok, report = False, {"total": 0, "approved": 0, "ratio": 0.0, "details": [], "min_ratio": min_ratio}
    else:
        ok, report = evaluate_anytext_against_domains(TOP_DOMAINS, payload, min_ratio=min_ratio)

    status = "✅ PASS" if ok else "⚠️ FAIL"
    header = f"### Evaluation — Tavily Top Domains ({status})"
    summary = (f"- Total: {report['total']}\n"
               f"- Approved: {report['approved']}\n"
               f"- Ratio: {report['ratio']:.0%} (min {int(min_ratio*100)}%)\n")

    rows = (report.get("details") or [])[:10]
    lines = ["| Host | Approved | Title |", "|---|:---:|---|"]
    for r in rows:
        lines.append(f"| {r.get('host') or '-'} | {'✔' if r.get('approved') else '—'} | {r.get('title') or r.get('url') or '-'} |")

    note = "*Note: Evaluation compares extracted link domains to a fixed allow-list (`TOP_DOMAINS`) and does not re-query tools.*"
    return "\n".join([header, summary, *lines, note])


def evaluate_tavily_results(TOP_DOMAINS: set[str], raw: str, min_ratio: float = 0.4):
    """
    Evaluate whether plain-text research results mostly come from trusted domains.

    Args:
        TOP_DOMAINS (set[str]): Set of trusted domains (e.g., 'arxiv.org', 'nature.com').
        raw (str): Plain text or Markdown containing URLs.
        min_ratio (float): Minimum trusted ratio required to pass (e.g., 0.4 = 40%).

    Returns:
        tuple[bool, str]: (flag, markdown_report)
            flag -> True if PASS, False if FAIL
            markdown_report -> Markdown-formatted summary of the evaluation
    """
    # Extract URLs from the text
    url_pattern = re.compile(r'https?://[^\s\]\)>\}]+', flags=re.IGNORECASE)
    urls = url_pattern.findall(raw)

    if not urls:
        return False, """### Evaluation — Tavily Top Domains
No URLs detected in the provided text.
Please include links in your research results.
"""

    # Count trusted vs total
    total = len(urls)
    trusted_count = 0
    details = []

    for url in urls:
        domain = url.split("/")[2]
        trusted = any(td in domain for td in TOP_DOMAINS)
        if trusted:
            trusted_count += 1
        details.append(f"- {url} → {'✅ TRUSTED' if trusted else '❌ NOT TRUSTED'}")

    ratio = trusted_count / total if total > 0 else 0.0
    flag = ratio >= min_ratio

    # Markdown report
    report = f"""
### Evaluation — Tavily Top Domains
- Total results: {total}
- Trusted results: {trusted_count}
- Ratio: {ratio:.2%}
- Threshold: {min_ratio:.0%}
- Status: {"✅ PASS" if flag else "❌ FAIL"}

**Details:**
{chr(10).join(details)}
"""
    return flag, report


# ================================
# HTML Rendering and Logging Functions (from M5_UGL_1)
# ================================
def render_pretty_table_html(df: pd.DataFrame, title: str = "Data Table") -> str:
    table_html = df.to_html(index=False, classes="styled-table")
    return f"""
    <style>
      .styled-table {{
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 14px;
        width: 100%;
        color: black;
        box-shadow: 0 0 5px rgba(0,0,0,0.1);
      }}
      .styled-table th, .styled-table td {{
        border: 1px solid #ddd;
        padding: 8px;
      }}
      .styled-table th {{
        background-color: #007acc;
        color: white;
        text-align: left;
      }}
      .styled-table tr:nth-child(even) {{ background-color: #e6f4ff; }}
      .styled-table tr:nth-child(odd)  {{ background-color: white;    }}
    </style>
    <h3>{escape(title)}</h3>
    {table_html}
    """


def format_logs_as_pretty_html(logs: list[dict], logo_path: str = "dl_logo.jpg") -> str:
    status_styles = {
        "success": {"bg": "#e0f0ff", "color": "#000000"},
        "fixed":   {"bg": "#fffbe6", "color": "#333333"},
        "error":   {"bg": "#ffe6e6", "color": "#000000"},
    }
    card_blocks = ""
    for log in logs:
        status = log.get("status", "success")
        style = status_styles.get(status, {"bg": "#f4f4f4", "color": "#000000"})
        bg, text_color = style["bg"], style["color"]
        step = escape(str(log.get("step", "")))
        desc = escape(str(log.get("description", "")))
        stxt = escape(str(status))
        card_blocks += f"""
        <div style="display:flex;align-items:center;background-color:{bg};margin:12px 0;
                    padding:12px 16px;border-radius:8px;box-shadow:2px 2px 5px rgba(0,0,0,0.05);">
          <img src="https://coursera-university-assets.s3.amazonaws.com/b4/5cb90bb92f420b99bf323a0356f451/Icon.png"
               alt="Logo" style="height:60px;margin-right:16px;border-radius:6px;"/>
          <div style="color:{text_color};">
            <h3 style="margin:0 0 4px 0;">Step {step}</h3>
            <p style="margin:4px 0;font-size:14px;">{desc}</p>
            <p style="margin:4px 0;"><strong>Status:</strong> <code>{stxt}</code></p>
          </div>
        </div>
        """
    return f"""
    <div style="font-family:Arial,sans-serif;max-width:800px;margin:auto;">
      <div style="text-align:center;padding:20px 0;">
        <img src="https://learn.deeplearning.ai/assets/dlai-logo.png" alt="Logo" style="max-height:80px;"/>
        <h2 style="margin-top:10px;">Customer Return Workflow Summary</h2>
      </div>
      {card_blocks}
    </div>
    """


def render_image_with_quote_html(image_url: str, quote: str, width: int = 512) -> None:
    html = f"""
    <div style="position:relative;width:{width}px;margin-bottom:20px;">
      <img src="{escape(image_url)}" style="width:100%;border-radius:8px;display:block;">
      <div style="
          position:absolute;bottom:20px;left:50%;transform:translateX(-50%);
          background:rgba(0,0,0,0.6);color:white;padding:10px 20px;border-radius:8px;
          font-size:1.2em;font-family:'Segoe UI',sans-serif;font-weight:500;text-align:center;
          text-shadow:1px 1px 4px #000;">
        {escape(quote)}
      </div>
    </div>
    """
    display(HTML(html))


def log_tool_call_html(tool_name: str, arguments: Any) -> None:
    display(HTML(f"""
      <div style="border-left:4px solid #1976D2;padding:.8em;margin:1em 0;
                  background-color:#e3f2fd;color:#0D47A1;font-family:'Segoe UI',sans-serif;">
        <div style="font-size:15px;font-weight:bold;margin-bottom:4px;">
          📞 <span style="color:#0B3D91;">Tool Call:</span> <span style="color:#0B3D91;">{escape(str(tool_name))}</span>
        </div>
        <code style="display:block;background:#e8f0fe;color:#1b1b1b;padding:6px;border-radius:4px;
                     font-size:13px;white-space:pre-wrap;">{escape(str(arguments))}</code>
      </div>
    """))


def log_tool_result_html(result: Any) -> None:
    display(HTML(f"""
      <div style="border-left:4px solid #558B2F;padding:.8em;margin:1em 0;
                  background-color:#f1f8e9;color:#33691E;">
        <strong>✅ Tool Result:</strong>
        <pre style="white-space:pre-wrap;font-size:13px;color:#2E7D32;">{escape(str(result))}</pre>
      </div>
    """))


def log_final_summary_html(content: str) -> None:
    display(HTML(f"""
      <div style="border-left:4px solid #2E7D32;padding:1em;margin:1em 0;
                  background-color:#e8f5e9;color:#1B5E20;">
        <strong>✅ Final Summary:</strong>
        <pre style="white-space:pre-wrap;font-size:13px;color:#1B5E20;">{escape(content.strip())}</pre>
      </div>
    """))


def log_unexpected_html() -> None:
    display(HTML("""
      <div style="border-left:4px solid #F57C00;padding:1em;margin:1em 0;
                  background-color:#fff3e0;color:#E65100;">
        <strong>⚠️ Unexpected:</strong> No tool_calls or content returned.
      </div>
    """))


def log_agent_title_html(title: str, icon: str = "🕵️‍♂️") -> None:
    display(HTML(f"""
      <div style="padding:1em;margin:1em 0;background-color:#f0f4f8;border-left:6px solid #1976D2;">
        <h2 style="margin:0;color:#0D47A1;font-family:'Segoe UI',sans-serif;">
          {escape(icon)} {escape(title)}
        </h2>
      </div>
    """))


def handle_tool_calls_with_multiple_tools(response, messages, client, mcp_client=None, tools=None, tools_dict=None, max_iterations=5):
    """处理工具调用的循环，支持多个工具"""
    if mcp_client and not tools:
        tools = mcp_client.list_tools()
    for iteration in range(max_iterations):
        if response.choices[0].message.tool_calls:
            for tool_call in response.choices[0].message.tool_calls:
                print(tool_call)
                # 处理第一个工具调用
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments
                if tool_args:
                    args = json.loads(tool_args)
                else:
                    args = {}
                if mcp_client:
                    result = mcp_client.call_tool(tool_name, args)
                else:
                    result = tools_dict[tool_name](args)
                print(f"工具 {tool_name} 参数{args} 执行结果: {result}")

                # 将工具结果返回给LLM
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [tool_call.model_dump()]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })

                # 获取下一轮响应
                response = client.chat.completions.create(
                    model=response.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                )
        else:
            break
    print(iteration)
    return response, messages