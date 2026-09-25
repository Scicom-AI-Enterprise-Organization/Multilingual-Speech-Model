"""Synthetic packs for smoke-testing qwen3_mel_adamw.py without any real data.

Builds one small ChiniDataset pack per task, with the *real* tokenizer so every id is
the id training would see:

  multipacking-tts      <|im_start|>{text}<|speech_start|>{<|s_N|>}<|im_end|>
  multipacking-stt      <|im_start|><|STT|>{<|s_N|>}<|{lang}|>{text}<|im_end|>
  multipacking-stt-mel  <|im_start|><|STT|><|mel_start|>{<|mel|>}<|mel_end|><|{lang}|>{text}<|im_end|>

Text and speech tokens are random ids; the mel pack writes real synthesised audio files
(harmonic + envelope + noise) into <out>/audio and stores paths, exactly as the real pack
does, so the whole read -> resample -> STFT -> project -> scatter path runs for real.

The point is the plumbing, not the loss: it proves the three document types flow, the
mel scatter lands, and — with more than one rank — that a micro-batch holding no mel
document does not hang the all-reduce. Weight the mel pack *down* in --train_file to
make those mel-free micro-batches common.

    python dryrun_pack.py --out /root/share/mel-dryrun
    torchrun --nproc_per_node 2 -m qwen3_mel_adamw \\
      --model_name_or_path Qwen/Qwen3-0.6B-Base \\
      --stt_tokens_file /root/share/mel-dryrun/stt_added_tokens.json \\
      --audio_dir /root/share/mel-dryrun/audio \\
      --train_file "/root/share/mel-dryrun/multipacking-tts:1.0,/root/share/mel-dryrun/multipacking-stt:1.0,/root/share/mel-dryrun/multipacking-stt-mel:0.3" \\
      ...
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'stt'))

from mel_audio import MEL_TOKENS, SAMPLE_RATE, build_speech_tokenizer, mel_positions, trim_to_position

BLOCK_SIZE = 1024 * 10
TOKEN_COLUMNS = {
    'input_ids': 'uint32[]',
    'position_ids': 'uint32[]',
    'attention_mask': 'uint32[]',
    'audio': 'str',
    'text': 'str',
}
# the base vocab range random "text" is drawn from — above the special tokens, below
# the 65,536 speech tokens that get appended
TEXT_LO, TEXT_HI = 1000, 100000


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def speech_like(seconds, rng):
    """A harmonic stack under an envelope — compresses like speech, unlike white noise."""
    n = trim_to_position(int(seconds * SAMPLE_RATE))
    t = np.arange(n) / SAMPLE_RATE
    f0 = rng.uniform(90, 240)
    y = sum(rng.uniform(0.15, 0.5) / k * np.sin(2 * np.pi * f0 * k * t) for k in (1, 2, 3, 4, 5))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(2.0, 5.0) * t)
    y = y * envelope + 0.01 * rng.standard_normal(n)
    return (y / (np.abs(y).max() + 1e-6) * 0.7).astype(np.float32)


def token_block(docs):
    return {
        'input_ids': np.concatenate(docs).astype(np.uint32),
        'position_ids': np.concatenate([np.arange(len(d)) for d in docs]).astype(np.uint32),
        'attention_mask': np.array([len(d) for d in docs], dtype=np.uint32),
        'audio': '',
        'text': '',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='/root/share/mel-dryrun')
    parser.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
    parser.add_argument('--tts-blocks', type=int, default=60)
    parser.add_argument('--stt-blocks', type=int, default=60)
    parser.add_argument('--mel-blocks', type=int, default=20)
    parser.add_argument('--languages', type=int, default=32)
    parser.add_argument('--min-seconds', type=float, default=1.5)
    parser.add_argument('--max-seconds', type=float, default=8.0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    import soundfile as sf

    from chinidataset import ParquetWriter
    from multipacking_stt_mel import COLUMNS as MEL_COLUMNS
    from multipacking_stt_mel import make_block as mel_block, probe_samples

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # written first so the tokenizer is built from it, exactly as training does
    languages = [f'<|lng{i:03d}_Latn|>' for i in range(args.languages)]
    stt_tokens = ['<|STT|>'] + languages
    stt_tokens_file = out / 'stt_added_tokens.json'
    with open(stt_tokens_file, 'w') as f:
        json.dump(stt_tokens, f, indent=2)

    log(f'building tokenizer from {args.model} (+65,537 speech, {len(stt_tokens)} stt, 3 mel)')
    tokenizer = build_speech_tokenizer(args.model, stt_tokens_file=str(stt_tokens_file),
                                       add_mel_tokens=True)
    tid = tokenizer.convert_tokens_to_ids
    ids = {
        'im_start': tid('<|im_start|>'), 'im_end': tid('<|im_end|>'),
        'speech_start': tid('<|speech_start|>'), 'stt': tid('<|STT|>'),
        's0': tid('<|s_0|>'), 'mel': tid('<|mel|>'),
        'mel_start': tid('<|mel_start|>'), 'mel_end': tid('<|mel_end|>'),
        'langs': [tid(t) for t in languages],
    }
    log(f'vocab {len(tokenizer)} | <|s_0|>={ids["s0"]} <|mel|>={ids["mel"]}')
    assert -1 not in ids.values() and None not in ids.values()

    def text_ids(n):
        return rng.integers(TEXT_LO, TEXT_HI, size=n).tolist()

    def speech_ids(n):
        return (ids['s0'] + rng.integers(0, 65536, size=n)).tolist()

    def duration():
        return float(rng.uniform(args.min_seconds, args.max_seconds))

    def fill(target_blocks, next_doc, writer, columns_have_audio=False):
        """Greedy 10,240-token fill, same as the real packs."""
        docs, paths, samples, count, written = [], [], [], 0, 0
        while written < target_blocks:
            doc = next_doc()
            seq = doc[0] if columns_have_audio else doc
            if count + len(seq) > BLOCK_SIZE and docs:
                writer.write(mel_block(docs, paths, samples) if columns_have_audio else token_block(docs))
                written += 1
                docs, paths, samples, count = [], [], [], 0
            docs.append(seq)
            count += len(seq)
            if columns_have_audio:
                paths.append(doc[1])
                samples.append(doc[2])
        return written

    # ---- TTS: text -> speech tokens
    log(f'packing {args.tts_blocks} TTS blocks')
    with ParquetWriter(out=str(out / 'multipacking-tts'), columns=TOKEN_COLUMNS, compression=None) as w:
        def tts_doc():
            secs = duration()
            return ([ids['im_start']] + text_ids(max(2, int(secs * 2.5)))
                    + [ids['speech_start']] + speech_ids(int(secs * 50)) + [ids['im_end']])
        fill(args.tts_blocks, tts_doc, w)

    # ---- STT: speech tokens -> text
    log(f'packing {args.stt_blocks} STT blocks')
    with ParquetWriter(out=str(out / 'multipacking-stt'), columns=TOKEN_COLUMNS, compression=None) as w:
        def stt_doc():
            secs = duration()
            return ([ids['im_start'], ids['stt']] + speech_ids(int(secs * 50))
                    + [int(rng.choice(ids['langs']))] + text_ids(max(2, int(secs * 2.5))) + [ids['im_end']])
        fill(args.stt_blocks, stt_doc, w)

    # ---- STT raw mel: audio files on disk -> text
    audio_dir = out / 'audio'
    audio_dir.mkdir(exist_ok=True)
    log(f'packing {args.mel_blocks} raw-mel blocks (FLAC files into {audio_dir})')
    audio_seconds = [0.0]
    written_files = [0]
    with ParquetWriter(out=str(out / 'multipacking-stt-mel'), columns=MEL_COLUMNS, compression=None) as w:
        def mel_doc():
            y = speech_like(duration(), rng)
            rel = f'{written_files[0]:06d}.flac'
            written_files[0] += 1
            sf.write(str(audio_dir / rel), y, SAMPLE_RATE, format='FLAC', subtype='PCM_16')
            # go back through the same header probe the real pack uses, so the placeholder
            # count comes from the file on disk rather than the array in memory
            n = probe_samples(str(audio_dir / rel))
            audio_seconds[0] += n / SAMPLE_RATE
            positions = mel_positions(n)
            seq = ([ids['im_start'], ids['stt'], ids['mel_start']] + [ids['mel']] * positions
                   + [ids['mel_end'], int(rng.choice(ids['langs']))]
                   + text_ids(max(2, positions // 20)) + [ids['im_end']])
            return seq, rel, n
        fill(args.mel_blocks, mel_doc, w, columns_have_audio=True)

    from chinidataset import StreamingDataset
    summary = {}
    for name in ('multipacking-tts', 'multipacking-stt', 'multipacking-stt-mel'):
        summary[name] = len(StreamingDataset(local=str(out / name)))
    summary['mel_audio_hours'] = round(audio_seconds[0] / 3600, 3)
    summary['mel_audio_files'] = written_files[0]
    summary['vocab_size'] = len(tokenizer)
    with open(out / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    log(f'done: {summary}')


if __name__ == '__main__':
    main()
