"""Fetch a SMALL sample of real ModelNet10 meshes for the local tests, without downloading the 473 MB zip.

It reads the zip's file index from the official Princeton server with HTTP Range requests, then fetches only the
chosen files (default: 5 train + 2 test per class = 70 meshes, ~6 MB transferred, ~27 MB unpacked).

The meshes are NOT ours to redistribute (academic-use dataset): dev/sample/ is git-ignored. Keep it that way.

Usage (from the project root):  python dev/fetch_sample.py [--out dev/sample] [--train 5] [--test 2]
Then:                           python dev/run_local_test.py --sample-dir dev/sample --workdir <new empty dir>
"""
import argparse, collections, struct, urllib.request, zlib
from pathlib import Path

URL = "https://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"   # official link (see README)


def get(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "pointnet-sample-fetch"})
    with urllib.request.urlopen(req, timeout=120) as r:
        if r.status != 206:
            raise RuntimeError(f"server ignored the Range request (HTTP {r.status}); cannot fetch a sample")
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "sample"))
    ap.add_argument("--train", type=int, default=5, help="meshes per class from the official train split")
    ap.add_argument("--test", type=int, default=2, help="meshes per class from the official test split")
    args = ap.parse_args()

    with urllib.request.urlopen(urllib.request.Request(URL, method="HEAD"), timeout=60) as r:
        size = int(r.headers["Content-Length"])
    tail = get(URL, size - 65536, size - 1)                       # the zip index ("central directory") sits at the end
    i = tail.rfind(b"PK\x05\x06")
    cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
    cd = get(URL, cd_off, cd_off + cd_size - 1)

    entries, p = [], 0
    while cd[p:p + 4] == b"PK\x01\x02":
        method, = struct.unpack("<H", cd[p + 10:p + 12])
        csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
        nlen, elen, clen = struct.unpack("<HHH", cd[p + 28:p + 34])
        offset, = struct.unpack("<I", cd[p + 42:p + 46])
        name = cd[p + 46:p + 46 + nlen].decode()
        if name.startswith("ModelNet10/") and name.endswith(".off") and not name.split("/")[-1].startswith("._"):
            entries.append((name, method, csize, usize, offset))
        p += 46 + nlen + elen + clen

    groups = collections.defaultdict(list)
    for e in entries:
        _, cls, split, _ = e[0].split("/")
        groups[(cls, split)].append(e)
    want = {"train": args.train, "test": args.test}
    out, n, total = Path(args.out), 0, 0
    for (cls, split), lst in sorted(groups.items()):
        lst.sort()
        k = min(want[split], len(lst))
        picks = sorted({round(j * (len(lst) - 1) / max(1, k - 1)) for j in range(k)})   # evenly spaced, deterministic
        for j in picks:
            name, method, csize, usize, offset = lst[j]
            blob = get(URL, offset, offset + 30 + 1024 + csize)
            nlen, elen = struct.unpack("<HH", blob[26:30])
            data = blob[30 + nlen + elen:30 + nlen + elen + csize]
            raw = zlib.decompress(data, -15) if method == 8 else data
            assert len(raw) == usize, f"size mismatch for {name}"
            dest = out / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
            n += 1; total += len(raw)
    print(f"wrote {n} meshes ({total / 1e6:.1f} MB) under {out / 'ModelNet10'}")


if __name__ == "__main__":
    main()
