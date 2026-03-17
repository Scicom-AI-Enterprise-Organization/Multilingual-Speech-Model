import os
import sys
import subprocess
import soundfile as sf
import copy
import itertools
import librosa
import re
from pathlib import Path
from huggingface_hub import snapshot_download
from functools import partial
from multiprocess import Pool
from datasets import load_dataset
from tqdm import tqdm
import click

REPO_URL = "https://github.com/fishaudio/fish-speech.git"
REPO_DIR = "fish-speech"

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
    
def old_chunks(l, n):
    for i in range(0, len(l), n):
        yield (l[i: i + n], i // n)

def chunks(l, devices):
    chunk_size = len(l) // len(devices)
    remainder = len(l) % len(devices)
    start = 0
    for i in range(len(devices)):
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        yield (l[start:end], devices[i])
        start = end

def multiprocessing(strings, function, cores=6, returned=True):
    df_split = old_chunks(strings, len(strings) // cores)
    pool = Pool(cores)
    pooled = pool.map(function, df_split)
    pool.close()
    pool.join()

    if returned:
        return list(itertools.chain(*pooled))

def check(indices_device_pair):
    rows, device = indices_device_pair
    filtered = []
    for r in tqdm(rows):
        filename = os.path.join(r['output'], f"{r['index']}-{r['retry']}.mp3")
        try:
            sf.read(filename)
            continue
        except:
            pass
        filtered.append(r)

    return filtered

def loop(indices_device_pair):
    rows, device = indices_device_pair
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    import torch
    torch.set_float32_matmul_precision('high')

    sys.path.insert(0, os.path.join(os.getcwd(), 'fish-speech'))
    from fish_speech.models.text2semantic.inference import init_model, generate_long

    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = torch.bfloat16

    checkpoint_dir = snapshot_download(repo_id="fishaudio/s2-pro")

    llama_model, decode_one_token = init_model(
        checkpoint_path=checkpoint_dir,
        device=device,
        precision=precision,
        compile=False,
    )

    with torch.device(device):
        llama_model.setup_caches(
            max_batch_size=1,
            max_seq_len=llama_model.config.max_seq_len,
            dtype=next(llama_model.parameters()).dtype,
        )

    def load_codec(codec_checkpoint_path, target_device, target_precision):
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(Path("fish-speech/fish_speech/configs/modded_dac_vq.yaml"))
        codec = instantiate(cfg)

        state_dict = torch.load(codec_checkpoint_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if any("generator" in k for k in state_dict):
            state_dict = {
                k.replace("generator.", ""): v
                for k, v in state_dict.items()
                if "generator." in k
            }

        codec.load_state_dict(state_dict, strict=False)
        codec.eval()
        codec.to(device=target_device, dtype=target_precision)
        return codec

    codec_model = load_codec(os.path.join(checkpoint_dir, "codec.pth"), device, precision)

    @torch.no_grad()
    def encode_reference_audio(audio_path):
        wav_np, _ = librosa.load(audio_path, sr=codec_model.sample_rate, mono=True)
        wav = torch.from_numpy(wav_np).to(device)
        model_dtype = next(codec_model.parameters()).dtype
        audios = wav[None, None, :].to(dtype=model_dtype)
        audio_lengths = torch.tensor([wav.shape[0]], device=device, dtype=torch.long)
        indices, feature_lengths = codec_model.encode(audios, audio_lengths)
        return indices[0, :, : feature_lengths[0]]

    @torch.no_grad()
    def decode_codes_to_audio(merged_codes):
        audio = codec_model.from_indices(merged_codes[None])
        return audio[0, 0]

    for r in tqdm(rows):
        clone_from_audio = os.path.join("common-voice", r['language'], 'audio', r['audio_filename'])
        filename = os.path.join(r['output'], f"{r['index']}-{r['retry']}.mp3")

        try:
            sf.read(filename)
            continue
        except:
            pass

        try:
            prompt_tokens_list = [encode_reference_audio(clone_from_audio).cpu()]
            generator = generate_long(
                model=llama_model,
                device=device,
                decode_one_token=decode_one_token,
                text=r['target_text'],
                num_samples=1,
                max_new_tokens=1024,
                top_p=0.7,
                top_k=30,
                temperature=0.7,
                repetition_penalty=1.2,
                compile=False,
                iterative_prompt=True,
                chunk_length=200,
                prompt_text=[r['source_text']],
                prompt_tokens=prompt_tokens_list,
            )
            codes = []
            for response in generator:
                if response.action == "sample":
                    codes.append(response.codes)
                elif response.action == "next":
                    break

            merged_codes = codes[0] if len(codes) == 1 else torch.cat(codes, dim=1)
            merged_codes = merged_codes.to(device)

            audio_waveform = decode_codes_to_audio(merged_codes)
            audio_np = audio_waveform.cpu().float().numpy()
            
            os.makedirs(os.path.split(filename)[0], exist_ok = True)
            sf.write(filename, audio_np, codec_model.sample_rate)
        except Exception as e:
            print(e)
    
@click.command()
@click.option('--output')
@click.option('--replication', default = 1)
@click.option('--retry', default = 2)
def main(output, replication, retry):
    devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if devices is None:
        
        import torch
        devices = list(range(torch.cuda.device_count()))
    else:
        devices = [d.strip() for d in devices.split(',')]

    devices = replication * devices

    ds = load_dataset("Scicom-intl/Evaluation-Multilingual-VC", 'combine_filtered_whisper_large_v3')
    df = ds['train'].to_pandas()
    df['index'] = df.index
    rows = df.to_dict(orient='records')
    actual_rows = []
    for r in rows:
        for k in range(retry):
            r = copy.copy(r)
            r['retry'] = k
            r['output'] = output
            actual_rows.append(r)
    
    filtered = multiprocessing(actual_rows, check, cores=20)
    print(len(filtered))
    if len(filtered):
        df_split = list(chunks(filtered, devices))

        loop_partial = partial(loop)

        with Pool(len(devices)) as pool:
            pooled = pool.map(loop_partial, df_split)

if __name__ == '__main__':
    main()
