"""Download the small LIBERO-PRO BDDL/state files required by its README."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import urllib.parse
import urllib.request

root = Path(os.environ['PIGEY_ROOT'])
api = 'https://huggingface.co/api/datasets/zhouxueyang/LIBERO-Pro'


def read_url(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


revision_file = root / 'libero-pro-assets.revision'
revision = revision_file.read_text().strip() if revision_file.exists() else json.loads(read_url(api))['sha']
resolve = f'https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro/resolve/{revision}/'
revision_file.write_text(revision + '\n')
files = []
for directory in ('bddl_files', 'init_files'):
    url = f'{api}/tree/{revision}/{directory}?recursive=true&limit=1000'
    with urllib.request.urlopen(url, timeout=60) as response:
        entries = json.load(response)
        if response.headers.get('Link'):
            raise RuntimeError('Asset listing is paginated; update downloader before proceeding')
    files.extend(entry for entry in entries if entry['type'] == 'file')
print(f'LIBERO-PRO assets: {revision}; {len(files)} files, {sum(x["size"] for x in files)} bytes', flush=True)


def install(entry):
    relative = PurePosixPath(entry['path'])
    if relative.is_absolute() or '..' in relative.parts or relative.parts[0] not in ('bddl_files', 'init_files'):
        raise ValueError(relative)
    destination = root / 'LIBERO-PRO/libero/libero' / str(relative)
    def matches(content):
        if len(content) != entry['size']:
            return False
        if 'lfs' in entry:
            return hashlib.sha256(content).hexdigest() == entry['lfs']['oid']
        header = f'blob {len(content)}\0'.encode()
        return hashlib.sha1(header + content).hexdigest() == entry['oid']

    content = destination.read_bytes() if destination.exists() else b''
    if matches(content):
        return f'{hashlib.sha256(content).hexdigest()}  {relative}\n'
    content = read_url(resolve + urllib.parse.quote(str(relative)))
    if not matches(content):
        raise RuntimeError(f'Asset content hash mismatch: {relative}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.download')
    temporary.write_bytes(content)
    temporary.replace(destination)
    return f'{hashlib.sha256(content).hexdigest()}  {relative}\n'


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    checksums = list(executor.map(install, files))
(root / 'libero-pro-assets.SHA256SUMS.txt').write_text(''.join(checksums))
print('LIBERO-PRO task assets installed; pinned Git/LFS hashes verified and SHA-256 manifest saved.', flush=True)
