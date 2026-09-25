"""Hub downloads that survive the CDN dropping a connection.

`us.aws.cdn.hf.co` regularly breaks a transfer mid-file with ChunkedEncodingError or a
read timeout. Both `snapshot_download` and `hf_hub_download` resume from what is already
on disk, so a retry costs only the files that were still in flight — but neither retries
on its own, and a multi-TB ingest will hit this many times.

    from hf_retry import snapshot_with_retry
    snapshot_with_retry('malaysia-ai/Multilingual-TTS', local_dir=..., allow_patterns=[...])
"""

import os
import time

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

DEFAULT_ATTEMPTS = 10
DEFAULT_BACKOFF = 30


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def _with_retry(fn, what, attempts, backoff, **kwargs):
    for attempt in range(attempts):
        try:
            return fn(**kwargs)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if attempt == attempts - 1:
                log(f'{what}: giving up after {attempts} attempts')
                raise
            wait = backoff * min(attempt + 1, 4)      # 30s, 60s, 90s, then 120s
            log(f'{what}: {type(e).__name__}: {str(e)[:120]} — retry {attempt + 1}/{attempts - 1} in {wait}s')
            time.sleep(wait)


def snapshot_with_retry(repo_id, attempts=DEFAULT_ATTEMPTS, backoff=DEFAULT_BACKOFF, **kwargs):
    from huggingface_hub import snapshot_download

    kwargs.setdefault('repo_type', 'dataset')
    return _with_retry(snapshot_download, f'snapshot {repo_id}', attempts, backoff,
                       repo_id=repo_id, **kwargs)


def file_with_retry(repo_id, filename, attempts=DEFAULT_ATTEMPTS, backoff=DEFAULT_BACKOFF, **kwargs):
    from huggingface_hub import hf_hub_download

    kwargs.setdefault('repo_type', 'dataset')
    return _with_retry(hf_hub_download, f'{repo_id}:{filename}', attempts, backoff,
                       repo_id=repo_id, filename=filename, **kwargs)
