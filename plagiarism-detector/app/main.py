import hashlib
import hmac
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import Cookie, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .engine import Corpus, check, extract_text

BASE = Path(__file__).parent
import os
from pathlib import Path

DB_PATH = Path(os.environ.get("DB_PATH", "/tmp/plagiarism.db"))
MAX_BYTES = 10 * 1024 * 1024
SESSION_DAYS = 7
PASSWORD_ITERATIONS = 600_000

app = FastAPI(title="Plagiarism Detection System")


@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def now_utc():
    return datetime.now(timezone.utc)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS
    )
    return f"pbkdf2_sha256${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, salt_hex, expected = encoded.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex),
            PASSWORD_ITERATIONS,
        ).hex()
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


def create_session(user_id: int, response: Response, request: Request):
    token = secrets.token_urlsafe(32)
    expires_at = now_utc() + timedelta(days=SESSION_DAYS)
    token_hash = hashlib.sha256(token.encode("ascii")).hexdigest()
    with db() as c:
        c.execute(
            "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
            (token_hash, user_id, expires_at.isoformat()),
        )
    response.set_cookie(
        "session", token, max_age=SESSION_DAYS * 24 * 60 * 60, path="/",
        httponly=True, secure=request.url.scheme == "https", samesite="strict",
    )


def current_user(session: str | None = Cookie(default=None)):
    if not session:
        raise HTTPException(401, "Please sign in to continue.")
    token_hash = hashlib.sha256(session.encode("ascii", "ignore")).hexdigest()
    with db() as c:
        user = c.execute(
            """SELECT users.id, users.name, users.email
               FROM sessions JOIN users ON users.id = sessions.user_id
               WHERE sessions.token_hash = ? AND sessions.expires_at > ?""",
            (token_hash, now_utc().isoformat()),
        ).fetchone()
    if user is None:
        raise HTTPException(401, "Your session has expired. Please sign in again.")
    return dict(user)


def user_corpus(user_id: int) -> Corpus:
    corpus = Corpus()
    with db() as c:
        for row in c.execute(
            "SELECT id, title, text FROM documents WHERE owner_id = ?", (user_id,)
        ):
            corpus.add(row["id"], row["title"], row["text"])
    return corpus


@app.on_event("startup")
def startup():
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            expires_at TEXT NOT NULL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL, text TEXT NOT NULL,
            owner_id INTEGER REFERENCES users(id))""")
        columns = {row["name"] for row in c.execute("PRAGMA table_info(documents)")}
        if "owner_id" not in columns:
            c.execute("ALTER TABLE documents ADD COLUMN owner_id INTEGER REFERENCES users(id)")


@app.post("/api/auth/register")
def register(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
):
    name = name.strip()
    email = email.strip().lower()
    if not name or len(name) > 100:
        raise HTTPException(400, "Enter a name up to 100 characters.")
    if len(email) > 254 or "@" not in email:
        raise HTTPException(400, "Enter a valid email address.")
    if len(password) < 12:
        raise HTTPException(400, "Use a password with at least 12 characters.")
    if len(password) > 256:
        raise HTTPException(400, "Password must be 256 characters or fewer.")
    with db() as c:
        try:
            user_id = c.execute(
                "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                (name, email, hash_password(password)),
            ).lastrowid
        except sqlite3.IntegrityError:
            raise HTTPException(409, "An account with that email already exists.") from None
        if c.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1:
            c.execute("UPDATE documents SET owner_id = ? WHERE owner_id IS NULL", (user_id,))
    response = JSONResponse({"name": name, "email": email})
    create_session(user_id, response, request)
    return response


@app.post("/api/auth/login")
def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
):
    email = email.strip().lower()
    if len(password) > 256:
        raise HTTPException(400, "Password must be 256 characters or fewer.")
    with db() as c:
        user = c.execute(
            "SELECT id, name, email, password_hash FROM users WHERE email = ?", (email,)
        ).fetchone()
    if user is None or not verify_password(password, user["password_hash"]):
        raise HTTPException(401, "Email or password is incorrect.")
    response = JSONResponse({"name": user["name"], "email": user["email"]})
    create_session(user["id"], response, request)
    return response


@app.get("/api/auth/me")
def who_am_i(user: dict = Depends(current_user)):
    return {"name": user["name"], "email": user["email"]}


@app.post("/api/auth/logout")
def logout(
    response: Response,
    session: str | None = Cookie(default=None),
    _: dict = Depends(current_user),
):
    if session:
        token_hash = hashlib.sha256(session.encode("ascii", "ignore")).hexdigest()
        with db() as c:
            c.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
    response.delete_cookie("session", path="/", httponly=True, samesite="strict")
    return {"ok": True}


async def read_input(text: str, file: UploadFile | None) -> str:
    if file is not None and file.filename:
        data = await file.read()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "File is larger than 10 MB.")
        try:
            text = extract_text(file.filename, data)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception:
            raise HTTPException(400, "Could not read that file. It may be corrupted.")
    text = (text or "").strip()
    if not text:
        raise HTTPException(400, "No text found. Paste text or upload a file with selectable text.")
    return text


@app.post("/api/check")
async def api_check(
    text: str = Form(""), file: UploadFile | None = File(None),
    user: dict = Depends(current_user),
):
    content = await read_input(text, file)
    corpus = user_corpus(user["id"])
    result = check(content, corpus)
    result["text"] = content
    result["reference_count"] = len(corpus.docs)
    return result


@app.get("/api/corpus")
def list_corpus(user: dict = Depends(current_user)):
    corpus = user_corpus(user["id"])
    return [{"id": i, "title": d["title"], "words": len(d["tokens"])}
            for i, d in corpus.docs.items()]


@app.post("/api/corpus")
async def add_corpus(title: str = Form(...), text: str = Form(""),
                     file: UploadFile | None = File(None),
                     user: dict = Depends(current_user)):
    content = await read_input(text, file)
    with db() as c:
        cur = c.execute(
            "INSERT INTO documents (title, text, owner_id) VALUES (?, ?, ?)",
            (title, content, user["id"]),
        )
        doc_id = cur.lastrowid
    return {"id": doc_id, "title": title}


@app.delete("/api/corpus/{doc_id}")
def delete_corpus(doc_id: int, user: dict = Depends(current_user)):
    with db() as c:
        cur = c.execute(
            "DELETE FROM documents WHERE id = ? AND owner_id = ?", (doc_id, user["id"])
        )
        if cur.rowcount == 0:
            raise HTTPException(404, "Reference document not found.")
    return {"deleted": doc_id}


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")
