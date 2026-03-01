"""Download Tiny Shakespeare dataset for Phase 1 training."""

import os
import urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.txt")


def main():
    if os.path.exists(OUT):
        print(f"Already exists: {OUT} ({os.path.getsize(OUT):,} bytes)")
        return
    print(f"Downloading Tiny Shakespeare...")
    urllib.request.urlretrieve(URL, OUT)
    print(f"Saved: {OUT} ({os.path.getsize(OUT):,} bytes)")


if __name__ == "__main__":
    main()
