"""Download the IDSSE dataset (7 Bundesliga games, CC BY 4.0) and verify every file.

Bassek, M., Rein, R., Weber, H. et al. "An integrated dataset of spatiotemporal and event data in
elite soccer." Sci Data 12, 195 (2025). https://doi.org/10.1038/s41597-025-04505-y
figshare article 28196177.

    python -m possession.fetch_idsse            # all 21 files, about 2.6 GB
    python -m possession.fetch_idsse --no-events

Files already present with the right MD5 are skipped, so it is safe to re-run after an
interruption. Uses figshare's public API and the download URLs it returns.
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request

from possession import EXTERNAL_DIR

ARTICLE_ID = 28196177
API = "https://api.figshare.com/v2/articles/{}/files?page_size=100".format(ARTICLE_ID)
USER_AGENT = "mtah2027-possession/0.1 (+https://github.com/SandeshM777/more-than-a-hack-2027)"
IDSSE_DIR = os.path.join(EXTERNAL_DIR, "idsse")
CHUNK = 1 << 20


def _get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}),
                                  timeout=120)


def file_list():
    with _get(API) as response:
        return json.load(response)


def md5_of(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(entry, out_dir):
    path = os.path.join(out_dir, entry["name"])
    if os.path.exists(path) and os.path.getsize(path) == entry["size"] \
            and md5_of(path) == entry["computed_md5"]:
        print("ok      {}".format(entry["name"]))
        return path
    partial = path + ".part"
    with _get(entry["download_url"]) as response, open(partial, "wb") as handle:
        done = 0
        for block in iter(lambda: response.read(CHUNK), b""):
            handle.write(block)
            done += len(block)
            if entry["size"] > 50 * CHUNK and done % (50 * CHUNK) < CHUNK:
                print("        {:>6.0f} / {:.0f} MB".format(done / 1e6, entry["size"] / 1e6),
                      flush=True)
    if md5_of(partial) != entry["computed_md5"]:
        os.remove(partial)
        raise RuntimeError("MD5 mismatch for {}".format(entry["name"]))
    os.replace(partial, path)
    print("fetched {}  ({:.1f} MB)".format(entry["name"], entry["size"] / 1e6), flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-events", action="store_true", help="skip the raw event XMLs")
    parser.add_argument("--out", default=IDSSE_DIR)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    entries = sorted(file_list(), key=lambda e: (e["size"], e["name"]))
    if args.no_events:
        entries = [e for e in entries if "_events_" not in e["name"]]
    print("{} files, {:.1f} MB total -> {}".format(
        len(entries), sum(e["size"] for e in entries) / 1e6, args.out), flush=True)
    for entry in entries:
        fetch(entry, args.out)
    print("done")


if __name__ == "__main__":
    sys.exit(main())
