# Plagiarism Detection System (MVP)

## Run (Windows)

Use Python 3.12 for the pinned scikit-learn version. From this project folder:

```powershell
# Create this environment once, if it does not already exist.
py -3.12 -m venv ..\..\venv312
```

Then install dependencies and start the app:

```powershell
..\..\venv312\Scripts\python.exe -m pip install -r requirements.txt
..\..\venv312\Scripts\python.exe -m uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000

Create an account or sign in. Each account has its own private reference library.
Go to **Reference library** to add documents, then use **Check a document** to
paste text or upload a file and click **Check for plagiarism**.

## How it works
- **Copied wording:** 5-word shingles are hashed and winnowed into fingerprints; shared fingerprints reveal copied passages.
- **Similar meaning:** TF-IDF cosine similarity between sentences flags light paraphrase.
- Score = share of your words covered by either kind of match.
- Passwords are stored as salted PBKDF2 hashes; sign-in uses an HTTP-only,
  same-site session cookie. This MVP has no password reset or email verification.

## Not included yet (from the PRD)
Roles, PDF report export, web search, scanned-PDF OCR, embedding-based semantic search.
