"""Build the corpus: download a public PDF, extract text with page markers.

Corpus is never committed. Only the URL is, so anyone can rebuild it.
Default document: HSBC Holdings plc Annual Report and Accounts (public filing).
Swap CORPUS_URL for any public annual report.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import requests
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CORPUS_URL = os.environ.get(
    "CORPUS_URL",
    "https://www.hsbc.com/-/files/hsbc/investors/hsbc-results/2024/annual/pdfs/"
    "hsbc-holdings-plc/sea-250219-annual-report-and-accounts-2024-hk.pdf",
)
MAX_PAGES = int(os.environ.get("MAX_PAGES", "120"))


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 100_000:
        print(f"[skip] cached {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    print(f"[get] {url}")
    resp = requests.get(url, timeout=120, stream=True, headers={"User-Agent": "eval-lab/1.0"})
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fh:
        for chunk in resp.iter_content(1024 * 256):
            fh.write(chunk)
    print(f"[ok] {dest} ({dest.stat().st_size/1e6:.1f} MB)")
    return dest


def extract(pdf: Path, out: Path, max_pages: int) -> Path:
    reader = PdfReader(str(pdf))
    total = len(reader.pages)
    print(f"[pdf] pages={total}, taking first {min(total, max_pages)}")
    buf: list[str] = []
    for i, page in enumerate(reader.pages[:max_pages], start=1):
        text = page.extract_text() or ""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        if len(text.strip()) < 200:  # skip image-only pages
            continue
        buf.append(f"[[PAGE {i}]]\n{text.strip()}\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(buf), encoding="utf-8")
    print(f"[ok] {out} ({out.stat().st_size/1e3:.0f} KB, {len(buf)} pages kept)")
    return out


def main():
    pdf = DATA / "source.pdf"
    txt = DATA / "corpus.txt"
    download(CORPUS_URL, pdf)
    extract(pdf, txt, MAX_PAGES)
    print("\nNow write evals/golden_rag.yaml against this corpus (see README section 4).")


if __name__ == "__main__":
    main()
