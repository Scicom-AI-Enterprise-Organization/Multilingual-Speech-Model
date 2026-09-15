"""Regression tests for the raw-mel STT path: python test_mel_pipeline.py

These pin the invariants that fail *silently* — a frame-rate or ordering mistake does
not crash, it trains a model on audio that belongs to a different utterance.

Needs torch, transformers, soundfile and soxr; no GPU and no dataset.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mel_audio import (
    HOP_LENGTH,
    LEAD,
    MEL_STACK,
    N_MELS,
    SAMPLE_RATE,
    SAMPLES_PER_POSITION,
    MelProjector,
    WhisperMel,
    load_segments,
    unpack_block,
    layout_segments,
    mel_positions,
    mixture_index,
    parse_train_files,
    trim_to_position,
)

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def tone(seconds, freq=220.0, amp=0.4, seed=0):
    n = trim_to_position(int(seconds * SAMPLE_RATE))
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SAMPLE_RATE
    y = amp * np.sin(2 * np.pi * freq * t) + 0.05 * rng.standard_normal(n)
    return y.astype(np.float32)


def run_mel(waveforms):
    buffer, frame_index, segment_id = layout_segments(waveforms)
    with torch.no_grad():
        return WhisperMel()(torch.from_numpy(buffer), torch.from_numpy(frame_index),
                            torch.from_numpy(segment_id), len(waveforms))


@test
def test_matches_whisper_feature_extractor():
    """The batched buffer must reproduce WhisperFeatureExtractor utterance by utterance."""
    from transformers import WhisperFeatureExtractor

    fe = WhisperFeatureExtractor(feature_size=N_MELS)
    waves = [tone(1.7, 220, seed=1), tone(4.3, 440, seed=2), tone(0.9, 130, seed=3)]
    got = run_mel(waves).numpy()

    off = 0
    for i, y in enumerate(waves):
        frames = len(y) // HOP_LENGTH
        ref = fe(y, sampling_rate=SAMPLE_RATE, return_tensors='np')['input_features'][0][:, :frames].T
        diff = np.abs(ref - got[off:off + frames]).max()
        assert diff < 1e-5, f'segment {i} differs from whisper by {diff}'
        off += frames
    assert off == got.shape[0]


@test
def test_no_cross_segment_bleed():
    """An utterance's mel must not change when its neighbours do."""
    target = tone(1.1, 300, seed=4)
    alone = run_mel([target]).numpy()
    loud = tone(2.0, 3500, amp=0.99, seed=5)
    packed = run_mel([loud, target, loud]).numpy()
    frames = len(target) // HOP_LENGTH
    start = len(loud) // HOP_LENGTH
    diff = np.abs(alone - packed[start:start + frames]).max()
    assert diff == 0.0, f'neighbours leaked into the utterance ({diff})'


@test
def test_frame_rate_is_50hz():
    """Positions per second must equal the NeuCodec rate, and divide by MEL_STACK."""
    for seconds in (0.5, 1.0, 3.7, 30.0):
        n = trim_to_position(int(seconds * SAMPLE_RATE))
        positions = mel_positions(n)
        assert positions == n // SAMPLES_PER_POSITION
        assert abs(positions / (n / SAMPLE_RATE) - 50.0) < 1e-6, 'not 50 positions/s'
        assert (n // HOP_LENGTH) % MEL_STACK == 0, 'frame count must divide by the stack'
    assert trim_to_position(SAMPLES_PER_POSITION * 3 + 7) == SAMPLES_PER_POSITION * 3


@test
def test_placeholder_count_matches_projector_output():
    """<|mel|> placeholders packed == rows the projector emits. The whole thing hinges
    on this: a mismatch means every later utterance reads the wrong audio."""
    waves = [tone(s, 200 + 50 * i, seed=i) for i, s in enumerate((2.2, 0.7, 5.0))]
    mel = run_mel(waves)
    projector = MelProjector(hidden_size=64)
    with torch.no_grad():
        features = projector(mel)
    assert features.shape[0] == sum(mel_positions(len(y)) for y in waves)
    assert features.shape[1] == 64


@test
def test_masked_scatter_ordering():
    """Projected frames must land on the placeholders in utterance order."""
    mel_id, hidden = 7, 4
    input_ids = torch.tensor([[1, mel_id, mel_id, 2, 3, mel_id, 4]])
    slots = input_ids == mel_id
    features = torch.arange(3 * hidden, dtype=torch.float32).reshape(3, hidden)

    embeds = torch.zeros(1, input_ids.shape[1], hidden)
    out = embeds.masked_scatter(slots.unsqueeze(-1), features)

    for k, position in enumerate([1, 2, 5]):
        assert torch.equal(out[0, position], features[k]), f'slot {position} got the wrong frames'
    for position in (0, 3, 4, 6):
        assert torch.equal(out[0, position], torch.zeros(hidden))


@test
def test_file_roundtrip_preserves_position_count():
    """The pack probes headers; the dataset reads files. They must agree on the length."""
    import soundfile as sf

    from multipacking_stt_mel import probe_samples

    waves = [tone(1.3, 210, seed=11), tone(0.6, 900, seed=12)]
    with tempfile.TemporaryDirectory() as tmp:
        paths, samples = [], []
        for i, y in enumerate(waves):
            rel = f'{i}.flac'
            sf.write(str(Path(tmp) / rel), y, SAMPLE_RATE, format='FLAC', subtype='PCM_16')
            paths.append(rel)
            samples.append(probe_samples(str(Path(tmp) / rel)))

        # the header probe must reproduce what the pack trimmed to
        assert samples == [len(y) for y in waves], f'{samples} != {[len(y) for y in waves]}'

        back = load_segments(tmp, paths, samples)
        for original, loaded in zip(waves, back):
            assert len(loaded) == len(original), f'{len(loaded)} != {len(original)} samples'
            assert mel_positions(len(loaded)) == mel_positions(len(original))
        buffer, frame_index, _ = layout_segments(back)
        assert len(frame_index) == sum(len(y) // HOP_LENGTH for y in waves)


@test
def test_resampled_length_is_forced_to_the_packed_count():
    """A file at another sample rate must still yield exactly the packed placeholders."""
    import soundfile as sf

    from multipacking_stt_mel import probe_samples

    y48 = np.sin(2 * np.pi * 220 * np.arange(48000 * 2) / 48000).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        sf.write(str(Path(tmp) / 'a.wav'), y48, 48000, subtype='PCM_16')
        n = probe_samples(str(Path(tmp) / 'a.wav'))
        assert n % SAMPLES_PER_POSITION == 0
        assert n <= 2 * SAMPLE_RATE, 'the probe must floor, never overshoot the real length'
        loaded = load_segments(tmp, ['a.wav'], [n])[0]
        assert len(loaded) == n
        assert mel_positions(len(loaded)) == n // SAMPLES_PER_POSITION


@test
def test_layout_rejects_unaligned_and_tiny():
    try:
        layout_segments([np.zeros(SAMPLES_PER_POSITION * 2 + 1, dtype=np.float32)])
        raise AssertionError('unaligned utterance should be rejected')
    except ValueError:
        pass
    try:
        layout_segments([np.zeros(0, dtype=np.float32)])
        raise AssertionError('empty utterance should be rejected')
    except ValueError:
        pass
    # why the reflect pad never runs off the front of an utterance: the shortest
    # non-empty aligned length is one position, which is already longer than LEAD
    assert LEAD < SAMPLES_PER_POSITION
    layout_segments([np.zeros(SAMPLES_PER_POSITION, dtype=np.float32)])


@test
def test_zero_probe_reaches_every_parameter():
    """A micro-batch with no mel document must still put the projector in the graph.

    Without this, DDP with find_unused_parameters=false hangs: the ranks that drew a mel
    document all-reduce the projector's gradients and wait for the ranks that did not.
    """
    projector = MelProjector(hidden_size=16)
    probe = projector.zero_probe(torch.zeros(1))
    assert float(probe.detach()) == 0.0, 'the probe must not move the loss'
    probe.backward()
    for name, p in projector.named_parameters():
        assert p.grad is not None, f'{name} got no gradient — DDP would stall'
        assert torch.isfinite(p.grad).all(), f'{name} gradient is not finite'

    # and a real loss is left exactly as it was
    loss = torch.tensor(2.5)
    assert float((loss + projector.zero_probe(torch.zeros(1))).detach()) == 2.5


@test
def test_pack_to_batch_roundtrip():
    """Whole path: pack a block, read it back, and check the audio still lines up.

    This is the one that would catch a real desync — the k-th run of <|mel|>
    placeholders must belong to the k-th packed utterance, all the way through the
    codec and the shared STFT buffer.
    """
    import soundfile as sf

    from multipacking_stt_mel import make_block, probe_samples

    IM_START, STT, MEL_START, MEL_END, LANG, IM_END, MEL_ID = 1, 2, 3, 4, 5, 6, 99
    waves = [tone(2.2, 200, seed=21), tone(0.5, 800, seed=22), tone(4.1, 330, seed=23)]

    tmp = tempfile.mkdtemp()
    docs, paths, samples = [], [], []
    for i, y in enumerate(waves):
        rel = f'{i}.flac'
        sf.write(str(Path(tmp) / rel), y, SAMPLE_RATE, format='FLAC', subtype='PCM_16')
        n = probe_samples(str(Path(tmp) / rel))
        positions = mel_positions(n)
        docs.append([IM_START, STT, MEL_START] + [MEL_ID] * positions
                    + [MEL_END, LANG, 40, 41, IM_END])
        paths.append(rel)
        samples.append(n)
    block = make_block(docs, paths, samples)

    # pack invariants the trainer's collator relies on
    assert list(block['attention_mask']) == [len(d) for d in docs]
    assert int(block['attention_mask'].sum()) == len(block['input_ids'])
    assert block['position_ids'][0] == 0 and block['position_ids'][len(docs[0]) - 1] == len(docs[0]) - 1
    assert block['position_ids'][len(docs[0])] == 0, 'position_ids must reset per document'

    # reader side, exactly what DatasetFixed does
    import json as _json
    back = load_segments(tmp, _json.loads(block['audio']),
                         block['audio_samples'].astype(np.int64))
    assert len(back) == len(waves)

    # collator side
    input_ids = torch.tensor(block['input_ids'].astype(np.int64))[None]
    features = MelProjector(hidden_size=8)(run_mel(back))
    slots = input_ids == MEL_ID
    assert int(slots.sum()) == features.shape[0], 'placeholders and mel positions disagree'

    # and each document's run of placeholders is its own utterance, in order
    runs = [sum(1 for i in d if i == MEL_ID) for d in docs]
    assert runs == [mel_positions(len(y)) for y in back]

    labels = block['input_ids'].astype(np.int64).copy()
    labels[np.isin(labels, [MEL_START, MEL_ID, MEL_END])] = -100
    assert int((labels == -100).sum()) == sum(runs) + 2 * len(docs)
    assert labels[0] == IM_START and labels[-1] == IM_END


@test
def test_unpack_block_never_mutates_the_row():
    """ChiniDataset caches rows and returns the same dict for a repeated index.

    Popping `audio` off it strips the audio from every later read of that block — the
    placeholders keep their random embeddings and training quietly continues on nothing.
    Writing `waveforms` back is the loud version: a ragged array on the next read.
    """
    import soundfile as sf

    from multipacking_stt_mel import make_block, probe_samples

    MEL_ID = 99
    with tempfile.TemporaryDirectory() as tmp:
        y = tone(1.1, 240, seed=31)
        sf.write(str(Path(tmp) / 'a.flac'), y, SAMPLE_RATE, format='FLAC', subtype='PCM_16')
        n = probe_samples(str(Path(tmp) / 'a.flac'))
        doc = [1, 2, 3] + [MEL_ID] * mel_positions(n) + [4, 5, 6]
        row = make_block([doc], ['a.flac'], [n])

        before = {k: (v if isinstance(v, str) else np.asarray(v).copy()) for k, v in row.items()}
        first = unpack_block(row, 10240, audio_dir=tmp)
        assert first is not None and 'waveforms' in first

        assert set(row) == set(before), f'unpack_block mutated the row keys: {set(before) - set(row)}'
        assert 'waveforms' not in row, 'derived data leaked back into the cached row'
        assert row['audio'] == before['audio'], 'the audio column was stripped off the row'

        # the decisive case: read the very same row object again
        second = unpack_block(row, 10240, audio_dir=tmp)
        assert second is not None and 'waveforms' in second, 'audio vanished on re-read'
        assert len(second['waveforms']) == len(first['waveforms'])
        assert np.array_equal(second['waveforms'][0], first['waveforms'][0])
        assert int((second['input_ids'] == MEL_ID).sum()) == mel_positions(n)


@test
def test_mixture_index():
    pairs = mixture_index([100, 40], [1.0, 1.0])
    assert len(pairs) == 140
    assert (pairs[:100, 0] == 0).all() and (pairs[100:, 0] == 1).all()
    assert sorted(pairs[:100, 1]) == list(range(100))

    pairs = mixture_index([100, 40], [0.25, 3.0])
    assert int((pairs[:, 0] == 0).sum()) == 25
    assert int((pairs[:, 0] == 1).sum()) == 120
    rows = pairs[pairs[:, 0] == 0][:, 1]
    assert rows.max() >= 96, 'a sub-1.0 weight must stride, not take a prefix'
    assert len(set(pairs[pairs[:, 0] == 1][:, 1].tolist())) == 40, 'repeats must cover the set'


@test
def test_parse_train_files():
    assert parse_train_files('a,b') == [('a', 1.0), ('b', 1.0)]
    assert parse_train_files('a:2.0, b:0.5') == [('a', 2.0), ('b', 0.5)]
    assert parse_train_files('/share/x/tts:1.5') == [('/share/x/tts', 1.5)]
    assert parse_train_files('hf://user/set') == [('hf://user/set', 1.0)]


def main():
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f'  ok    {fn.__name__}')
        except Exception as e:
            failed += 1
            print(f'  FAIL  {fn.__name__}: {type(e).__name__}: {e}')
    print(f'\n{len(TESTS) - failed}/{len(TESTS)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
