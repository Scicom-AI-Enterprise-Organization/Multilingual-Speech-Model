"""Raw log-mel audio input for the Qwen3 speech models.

    audio -> whisper log-mel (100 fps, 128 bins) -> stack 2 frames -> MLP -> LLM

This is malaysia-ai/malaya's `session/audiollm` wiring with the whisper *encoder*
removed: only whisper's mel front end survives and the LLM itself does the acoustic
modelling. Frames are stacked 2-at-a-time to reach 50 positions/s -- the same rate as
the NeuCodec `<|s_NNNN|>` tokens -- so a mel-STT document and a token-STT document cost
the same context for the same audio.

Stacking rather than pooling is deliberate. A reshape is information preserving and is
exactly `Conv1d(128, H, kernel_size=2, stride=2)`; an average pool pre-commits to one
fixed mixing of the two frames and low-passes away the ~10ms cues (plosive bursts, stop
closures, onset edges) that separate phonemes. Qwen2-Audio can afford to pool because
its stride-2 pool sits *after* 32 transformer layers that already made neighbouring
positions redundant. With no encoder here, adjacent mel frames are not redundant.

The pack stores audio *paths*; the dataset reads the files and resamples in the
dataloader worker, and the STFT runs on the GPU inside the model forward. Doing the STFT
in the dataloader too costs ~15-30 CPU-seconds per micro-batch and starves the GPU.
"""

import json
import os

import numpy as np
import torch
from torch import nn

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160          # 10ms -> 100 frames/s
N_MELS = 128              # whisper-large-v3
MEL_STACK = 2             # 2 frames -> 50 positions/s, the NeuCodec rate

# samples per LLM position, and the granularity every packed utterance is trimmed to so
# that a segment's frame count is always an exact multiple of MEL_STACK
SAMPLES_PER_POSITION = HOP_LENGTH * MEL_STACK      # 320

# Appended to the tokenizer *after* <|speech_start|>, the 65,536 <|s_N|> tokens, <|STT|>
# and the language tokens -- see stt/README.md, speech-token ids must not move.
MEL_TOKENS = ['<|mel_start|>', '<|mel|>', '<|mel_end|>']

# Layout padding around each utterance inside the shared STFT buffer. LEAD reproduces
# the reflect pad WhisperFeatureExtractor gets from center=True; TRAIL reproduces the
# zeros it gets from padding the utterance out to 30s. Together they are a multiple of
# HOP_LENGTH, which keeps every segment's frame grid aligned.
LEAD = N_FFT // 2         # 200
TRAIL = 280
SEG_PAD = LEAD + TRAIL    # 480 == 3 * HOP_LENGTH


def mel_positions(n_samples):
    """LLM positions a waveform of `n_samples` occupies (== `<|mel|>` placeholders)."""
    return n_samples // SAMPLES_PER_POSITION


def trim_to_position(n_samples):
    """Largest length <= n_samples that yields a whole number of stacked positions."""
    return (n_samples // SAMPLES_PER_POSITION) * SAMPLES_PER_POSITION


# ---------------------------------------------------------------- tokenizer

def build_speech_tokenizer(model_name_or_path, stt_tokens_file=None, add_mel_tokens=False):
    """Qwen3 tokenizer + speech tokens, in the one order every pack/trainer must share.

    <|speech_start|>, 65,536 <|s_N|>, then (optionally) the <|STT|> + language tokens
    from an STT pack's `stt_added_tokens.json`, then (optionally) MEL_TOKENS. Appending
    only ever at the end is what keeps `<|s_N|>` ids identical to the TTS trainers.
    """
    from transformers import AddedToken, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    extra = [AddedToken('<|speech_start|>')]
    for i in range(65536):
        extra.append(AddedToken(f'<|s_{i}|>'))
    if stt_tokens_file:
        with open(stt_tokens_file) as f:
            stt_tokens = json.load(f)
        extra.extend(AddedToken(t) for t in stt_tokens)
    if add_mel_tokens:
        extra.extend(AddedToken(t) for t in MEL_TOKENS)
    tokenizer.add_tokens(extra)
    return tokenizer


# ---------------------------------------------------------------- audio -> buffer

def load_segments(audio_dir, paths, samples):
    """Read a packed block's utterances off disk as 16kHz mono float32.

    The pack stores paths, not audio, so decode + resample happen here, in the dataloader
    worker, and the mel follows on the GPU. `samples` is the length the pack derived each
    utterance's `<|mel|>` placeholder count from; the decoded signal is forced to exactly
    that, because a resampler that rounds a few samples differently would otherwise shift
    every later utterance in the block onto the wrong placeholders.
    """
    import soundfile as sf
    import soxr

    out = []
    for path, n in zip(paths, samples):
        y, sr = sf.read(os.path.join(audio_dir, path), dtype='float32', always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != SAMPLE_RATE:
            y = soxr.resample(y, sr, SAMPLE_RATE)
        n = int(n)
        if len(y) < n:
            y = np.pad(y, (0, n - len(y)))
        out.append(np.ascontiguousarray(y[:n], dtype=np.float32))
    return out


def unpack_block(row, sequence_length, audio_dir=None):
    """One packed row -> one training sample, without mutating the row.

    The copy is load-bearing. ChiniDataset's reader caches rows and hands back the *same
    dict object* for a repeated index, so popping `audio` off it strips the audio from
    every later read of that block: `paths` comes back None, the block's `<|mel|>`
    placeholders keep their random embeddings, and training quietly continues on nothing.
    Writing `waveforms` back into it is the loud version of the same bug — the next read
    of that block hits a ragged array in the dtype loop.
    """
    data = dict(row)
    paths = data.pop('audio', None)
    samples = data.pop('audio_samples', None)
    data.pop('text', None)
    data.pop('token_type_ids', None)

    for k in list(data):
        data[k] = np.asarray(data[k]).astype(np.int64)

    if data['attention_mask'].max() > sequence_length:
        return None

    if paths and samples is not None and len(samples):
        if not audio_dir:
            raise ValueError('this pack stores audio paths; pass --audio_dir')
        data['waveforms'] = load_segments(audio_dir, json.loads(paths),
                                          np.asarray(samples).astype(np.int64))
    return data


def layout_segments(waveforms):
    """Pack utterances into one STFT buffer plus the frame indices to read back.

    Each utterance is written as ``[reflect(200) | y | zeros(280)]``. A single
    ``center=False`` STFT over the buffer then reproduces WhisperFeatureExtractor's
    frames for every utterance exactly -- whisper reflect-pads the front (``center=True``)
    and zero-pads the tail (its pad-to-30s) -- while no frame of one utterance can reach
    into its neighbour. The alternative, padding every utterance to a common length,
    would cost ~500MB of zeros per micro-batch at 30s clips.

    Returns (buffer, frame_index, segment_id): `frame_index` selects this block's real
    frames out of the STFT in utterance order, `segment_id` maps each of them back to
    its utterance (needed for whisper's per-utterance max normalisation).
    """
    lengths = np.array([len(y) for y in waveforms], dtype=np.int64)
    if not len(lengths):
        return (np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    if (lengths % SAMPLES_PER_POSITION).any():
        raise ValueError('every packed utterance must be a multiple of '
                         f'{SAMPLES_PER_POSITION} samples, got {lengths[lengths % SAMPLES_PER_POSITION != 0][:4]}')
    if (lengths <= LEAD).any():
        raise ValueError(f'utterances must be longer than {LEAD} samples to reflect-pad')

    offsets = np.concatenate([[0], np.cumsum(lengths + SEG_PAD)])
    buffer = np.zeros(int(offsets[-1]), dtype=np.float32)
    for i, y in enumerate(waveforms):
        o = int(offsets[i])
        buffer[o:o + LEAD + len(y)] = np.pad(y, (LEAD, 0), mode='reflect')

    starts = offsets[:-1] // HOP_LENGTH
    frames = lengths // HOP_LENGTH
    frame_index = np.concatenate([np.arange(s, s + n) for s, n in zip(starts, frames)])
    segment_id = np.repeat(np.arange(len(lengths)), frames)
    return buffer, frame_index.astype(np.int64), segment_id.astype(np.int64)


# ---------------------------------------------------------------- mel + projection

class WhisperMel(nn.Module):
    """WhisperFeatureExtractor's log-mel, batched over a `layout_segments` buffer.

    Buffers are non-persistent so they never enter a checkpoint (and so
    `from_pretrained` never treats them as missing weights to re-initialise).
    """

    def __init__(self, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
                 sampling_rate=SAMPLE_RATE):
        super().__init__()
        from transformers.audio_utils import mel_filter_bank

        self.n_fft = n_fft
        self.hop_length = hop_length
        filters = mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=n_mels,
            min_frequency=0.0,
            max_frequency=sampling_rate / 2.0,
            sampling_rate=sampling_rate,
            norm='slaney',
            mel_scale='slaney',
        )
        self.register_buffer('mel_filters', torch.from_numpy(filters).float(), persistent=False)
        self.register_buffer('window', torch.hann_window(n_fft), persistent=False)

    def forward(self, waveform, frame_index, segment_id, num_segments):
        """-> [len(frame_index), n_mels] float32, whisper-normalised per utterance."""
        # forced to fp32: loading the model in bf16 casts these buffers with it, and a
        # bf16 STFT loses the quiet end of the spectrum that whisper's (max - 8) floor
        # then spreads across the whole utterance
        waveform = waveform.float()
        stft = torch.stft(waveform, self.n_fft, self.hop_length, window=self.window.float(),
                          center=False, return_complex=True)
        if stft.shape[-1] <= int(frame_index[-1]):
            raise RuntimeError(f'STFT produced {stft.shape[-1]} frames, need {int(frame_index[-1]) + 1}')
        magnitudes = stft.abs().pow(2)
        mel = (self.mel_filters.float().T @ magnitudes).T[frame_index]
        log_spec = mel.clamp(min=1e-10).log10()

        # whisper clamps to (max - 8) with the max taken over one utterance, so the
        # dynamic-range floor must be per segment, not per batch
        seg_max = torch.full((num_segments,), float('-inf'),
                             dtype=log_spec.dtype, device=log_spec.device)
        seg_max = seg_max.scatter_reduce(0, segment_id, log_spec.amax(dim=-1), reduce='amax')
        log_spec = torch.maximum(log_spec, (seg_max - 8.0)[segment_id].unsqueeze(-1))
        return (log_spec + 4.0) / 4.0


class MelProjector(nn.Module):
    """Stack `stack` mel frames and project them into the LLM's embedding space.

    Input is the flat [total_frames, n_mels] stream from `WhisperMel`; every segment
    contributes a multiple of `stack` frames, so one global reshape is safe and no
    utterance is ever mixed with its neighbour.

    A wider, overlapping front end (`Conv1d(n_mels, n_mels * stack, kernel_size=5,
    stride=stack, padding=2)` in place of the reshape) is the upgrade to try if this
    underfits -- it needs the per-segment frame counts to pad each segment separately,
    which is why `forward` is shaped around the flat stream rather than a [B, T, C]
    tensor.
    """

    def __init__(self, hidden_size, n_mels=N_MELS, stack=MEL_STACK):
        super().__init__()
        self.n_mels = n_mels
        self.stack = stack
        self.norm = nn.LayerNorm(n_mels * stack)
        self.fc1 = nn.Linear(n_mels * stack, hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, mel):
        x = mel.reshape(-1, self.n_mels * self.stack)
        return self.fc2(self.act(self.fc1(self.norm(x))))

    def zero_probe(self, reference):
        """A 0-valued scalar that every parameter in this module contributes to.

        DDP with `find_unused_parameters=False` needs every rank to produce a gradient
        for every parameter on every step. A micro-batch that happened to draw no mel
        document leaves this module out of the graph, and the all-reduce then waits
        forever on ranks that did have one. Add this to the loss on those steps.

        Same fix as malaya's `session/audiollm/qwen_audio_stage2.py` (`dummy_audio`:
        feed 0.02s of zeros, `masked_audio_features.sum() * 0.0`), minus the encoder --
        `WhisperMel` holds only non-persistent buffers, so this module is the whole
        parameterised audio path and one zero frame-pair reaches all of it.
        """
        return self(reference.new_zeros(self.stack, self.n_mels)).sum() * 0.0


# ---------------------------------------------------------------- data mixture

def mixture_index(sizes, weights):
    """Flat [(dataset, row)] index for a weighted mixture of packed datasets.

    `weight` is how many epochs of a dataset go into one training epoch: 1.0 is a plain
    concat at natural proportions, 3.0 repeats it three times, 0.25 takes an evenly
    spread quarter. Sub-1.0 weights stride rather than take a prefix -- packs are
    written subset by subset, so the first N blocks are not a sample of the corpus.
    """
    pairs = []
    for i, (n, weight) in enumerate(zip(sizes, weights)):
        count = max(1, int(round(n * weight)))
        if count <= n:
            rows = np.linspace(0, n, count, endpoint=False).astype(np.int64)
        else:
            rows = np.arange(count) % n
        pairs.append(np.stack([np.full(count, i, dtype=np.int64), rows], axis=1))
    return np.concatenate(pairs)


def parse_train_files(spec):
    """'dir:1.0,dir2' -> [(dir, 1.0), (dir2, 1.0)]."""
    out = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        path, sep, weight = part.rpartition(':')
        if sep:
            try:
                out.append((path, float(weight)))
                continue
            except ValueError:      # a ':' that is part of the path, not a weight
                pass
        out.append((part, 1.0))
    return out
