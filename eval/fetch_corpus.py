"""
Rebuild the eval corpus (eval/corpus/) used to (re)generate the golden set and to run
the retrieval evaluation. The arXiv PDFs are not committed (public repo), so this fetches
them; the beginner guide is copied from example_docs/. Idempotent.

Usage: .venv/bin/python eval/fetch_corpus.py
"""
import shutil
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS = REPO_ROOT / "eval" / "corpus"

# arXiv PDFs (name -> URL). llm_introduction.pdf is copied from example_docs/.
ARXIV = {
    "attention_is_all_you_need.pdf": "https://arxiv.org/pdf/1706.03762",  # Vaswani et al., 2017
    "bert.pdf": "https://arxiv.org/pdf/1810.04805",                       # Devlin et al., 2018
}


def main():
    CORPUS.mkdir(parents=True, exist_ok=True)

    # 1) local copy of the beginner guide (also the app's sample doc)
    src = REPO_ROOT / "example_docs" / "llm_introduction.pdf"
    dst = CORPUS / "llm_introduction.pdf"
    if dst.exists():
        print(f"{dst.name} already present")
    else:
        shutil.copy(src, dst)
        print(f"copied {dst.name} from example_docs/")

    # 2) arXiv downloads
    for name, url in ARXIV.items():
        target = CORPUS / name
        if target.exists():
            print(f"{name} already present")
            continue
        print(f"downloading {name} from {url} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(target, "wb") as f:
            f.write(resp.read())
        print(f"  saved {name} ({target.stat().st_size} bytes)")

    print("Corpus ready:", CORPUS)


if __name__ == "__main__":
    main()
