# ingest.py
import argparse
import sqlite3
import os
from pypdf import PdfReader
import re
from tqdm import tqdm

CHUNK_MIN = 100
CHUNK_MAX = 400

def text_from_pdf(path):
    text = []
    reader = PdfReader(path)
    for page in reader.pages:
        p = page.extract_text() or ""
        text.append(p)
    return "\n".join(text)

# naive paragraph splitter
def split_into_chunks(text):
    # split on two or more newlines or sentences; then merge small ones
    paras = [p.strip() for p in re.split(r"\n{2,}|(?<=\.)\s\n", text) if p.strip()]
    chunks = []
    curr = ""
    for p in paras:
        if not curr:
            curr = p
        elif len(curr) + len(p) < CHUNK_MAX:
            curr += "\n\n" + p
        else:
            chunks.append(curr)
            curr = p
    if curr:
        chunks.append(curr)
    # ensure min size: merge tiny chunks forward
    out = []
    buff = ""
    for c in chunks:
        if not buff:
            buff = c
        elif len(buff) < CHUNK_MIN:
            buff += "\n\n" + c
        else:
            out.append(buff)
            buff = c
    if buff:
        out.append(buff)
    return out


def main(pdf_dir, sqlite_path):
    os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY,
        doc TEXT,
        title TEXT,
        chunk TEXT,
        page_numbers TEXT
    )''')
    conn.commit()
    idx = 0
    for fname in tqdm(os.listdir(pdf_dir)):
        if not fname.lower().endswith('.pdf'):
            continue
        fpath = os.path.join(pdf_dir, fname)
        text = text_from_pdf(fpath)
        chunks = split_into_chunks(text)
        for i,c in enumerate(chunks):
            cur.execute('INSERT INTO chunks (doc, title, chunk, page_numbers) VALUES (?, ?, ?, ?)',
                        (fname, fname, c, str(i)))
            idx += 1
        conn.commit()
    conn.close()
    print(f"Wrote {idx} chunks to {sqlite_path}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pdf-dir', required=True)
    p.add_argument('--sqlite-path', required=True)
    args = p.parse_args()
    main(args.pdf_dir, args.sqlite_path)