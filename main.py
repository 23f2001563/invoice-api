import json
import os
import re
import urllib.request
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

app = FastAPI()


class ExtractRequest(BaseModel):
    text: str


class Invoice(BaseModel):
    vendor: str
    amount: float
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


# ---------- rule-based extraction ----------
ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
CODES = "USD|EUR|GBP|INR|JPY|CAD|AUD|CHF|CNY|SGD|AED"
SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP"}
SUFFIX = (
    r"(?:Ltd|Limited|Inc|LLC|GmbH|Corp|Corporation|Co|Company|PLC|Pty|AG|"
    r"Industries|Group|Holdings|Enterprises|Solutions|Technologies|Systems|"
    r"Services|Partners|Trading)\b\.?"
)
NAME_TOKEN = r"[A-Z][\w&'\-]*"
VENDOR_SUFFIX = re.compile(rf"{NAME_TOKEN}(?:\s+{NAME_TOKEN})*?\s+{SUFFIX}(?:\s+{SUFFIX})*")
VENDOR_LABEL = re.compile(
    r"(?i:vendor|supplier|billed by|issued by|invoice from|from|payee|pay to)"
    r"\s*[:\-]?\s*([A-Z][^\n,;:]{1,80}?)"
    r"(?=\s+(?:[Tt]otal|[Aa]mount|[Dd]ue|[Ii]nvoice|[Pp]lease|for|is|has|dated?)\b|[,;\n]|$)"
)
LEADING_NOISE = {"invoice", "from", "bill", "billed", "by", "to", "vendor", "supplier",
                 "issued", "pay", "payee", "please", "dear", "total", "due"}
STRONG = re.compile(
    r"(?i:\b(?:total\s+due|amount\s+due|balance\s+due|grand\s+total|total\s+amount|"
    r"amount\s+payable))\b[^\d]{0,40}?(" + NUM + ")"
)
WEAK = re.compile(
    r"(?i:\b(?:total|balance|amount|payable|owed))\b[^\d]{0,40}?(" + NUM + ")"
)
ADJ = re.compile(rf"(?i:({CODES}))\s*({NUM})|({NUM})\s*(?i:({CODES}))\b|([$€£])\s*({NUM})")


def find_vendor(text):
    m = VENDOR_SUFFIX.search(text)
    name = m.group(0) if m else None
    if not name:
        m = VENDOR_LABEL.search(text)
        name = m.group(1) if m else None
    if not name:
        return "Unknown"
    words = name.split()
    while len(words) > 1 and words[0].lower().strip(":") in LEADING_NOISE:
        words.pop(0)
    return " ".join(words).strip(" \t:;,")


def find_date(text):
    valid = []
    for m in ISO_DATE.finditer(text):
        try:
            datetime.strptime(m.group(0), "%Y-%m-%d")
        except ValueError:
            continue
        valid.append(m)
    if not valid:
        raise ValueError("no date")
    for m in valid:
        ctx = text[max(0, m.start() - 25):m.start()].lower()
        if re.search(r"due|pay|\bby\b|before|until|deadline", ctx) and not re.search(
            r"issue|invoice date|dated|created", ctx
        ):
            return m.group(0)
    for m in valid:  # no clear due-date cue: first date not labelled as an issue date
        ctx = text[max(0, m.start() - 25):m.start()].lower()
        if not re.search(r"issue|invoice date|dated|created", ctx):
            return m.group(0)
    return valid[-1].group(0)


def find_currency(text):
    m = re.search(rf"(?i)\b({CODES})\b", text)
    if m:
        return m.group(1).upper()
    for sym, code in SYMBOLS.items():
        if sym in text:
            return code
    low = text.lower()
    for word, code in (("dollar", "USD"), ("euro", "EUR"), ("pound", "GBP"), ("rupee", "INR")):
        if word in low:
            return code
    return "USD"


def to_float(s):
    return round(float(s.replace(",", "")), 2)


def find_amount(text, vendor):
    t = ISO_DATE.sub(" DATE ", text)
    if vendor and vendor != "Unknown":
        t = t.replace(vendor, " VENDOR ")
    t = re.sub(r"\b[A-Za-z]+(?:-[A-Za-z0-9]+)*-\d[\w-]*", " ", t)      # INV-2026-001
    t = re.sub(r"#\s*\d+", " ", t)                                      # #1234
    t = re.sub(r"(?i:invoice\s*(?:no\.?|number|num\.?)\s*:?\s*)[\w-]+", " ", t)
    for pat in (STRONG, WEAK):
        m = pat.search(t)
        if m:
            return to_float(m.group(1))
    m = ADJ.search(t)
    if m:
        return to_float(m.group(2) or m.group(3) or m.group(6))
    nums = re.findall(NUM, t)
    if nums:
        return max(to_float(n) for n in nums)
    raise ValueError("no amount")


def rule_extract(text):
    vendor = find_vendor(text)
    return {
        "vendor": vendor,
        "amount": find_amount(text, vendor),
        "currency": find_currency(text),
        "date": find_date(text),
    }


# ---------- optional local-LLM fallback (only runs if OLLAMA_URL is set) ----------
def llm_extract(text):
    url = os.getenv("OLLAMA_URL")
    if not url:
        return None
    payload = {
        "model": os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": (
                "Extract invoice fields. Reply with JSON only, keys: vendor (string), "
                "amount (number, total due), currency (3-letter uppercase code), "
                "date (payment due date, YYYY-MM-DD).")},
            {"role": "user", "content": text},
        ],
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(json.loads(resp.read())["message"]["content"])


@app.post("/extract", response_model=Invoice)
def extract(req: ExtractRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be empty")
    try:
        data = rule_extract(text)
    except ValueError:
        try:
            data = llm_extract(text)
        except Exception:
            data = None
    if not data:
        raise HTTPException(status_code=422, detail="could not extract invoice fields")
    try:
        return Invoice(**data)
    except (ValidationError, TypeError):
        raise HTTPException(status_code=422, detail="extracted fields failed validation")


@app.exception_handler(Exception)
async def never_500(request: Request, exc: Exception):
    return JSONResponse(status_code=422, content={"detail": "could not process input"})


@app.get("/")
def root():
    return {"status": "ok"}