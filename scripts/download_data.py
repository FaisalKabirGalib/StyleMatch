"""Download the original research file and record provenance without changing it."""
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html'
URL = 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Clothing_Shoes_and_Jewelry_5.json.gz'
path = ROOT / 'data/raw' / URL.rsplit('/', 1)[1]
path.parent.mkdir(parents=True, exist_ok=True)
if not path.exists():
    temporary = path.with_suffix('.part')
    with urlopen(URL, timeout=60) as response, temporary.open('wb') as target:
        while chunk := response.read(1024 * 1024):
            target.write(chunk)
    temporary.replace(path)
with gzip.open(path, 'rt') as stream:
    rows = sum(1 for line in stream if line.strip())
assert rows == 278677, f'Unexpected source row count: {rows}'
manifest = {
    'dataset': 'Amazon product data 2014 Clothing Shoes and Jewelry 5-core',
    'source_page': SOURCE, 'download_url': URL,
    'verified_at_utc': datetime.now(timezone.utc).isoformat(),
    'bytes': path.stat().st_size,
    'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest(),
    'reviews': rows,
    'license_note': 'Publicly downloadable research dataset. The linked source page does not specify a standalone permissive data license; do not describe it as CC0 or unrestricted open-source data.',
}
(ROOT / 'data/source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
