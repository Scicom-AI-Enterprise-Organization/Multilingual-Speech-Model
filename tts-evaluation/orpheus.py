import os
import soundfile as sf
import copy
import itertools
import re
from functools import partial
from multiprocess import Pool
from datasets import load_dataset
from tqdm import tqdm
import click

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

    from snac import SNAC
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = "canopylabs/orpheus-3b-0.1-ft"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").cuda()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.cuda()

    def tokenise_audio(waveform):
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        waveform = waveform.to(dtype=torch.float32)

        waveform = waveform.unsqueeze(0)

        with torch.inference_mode():
            codes = snac_model.encode(waveform.to('cuda'))

        all_codes = []
        for i in range(codes[0].shape[1]):
            all_codes.append(codes[0][0][i].item()+128266)
            all_codes.append(codes[1][0][2*i].item()+128266+4096)
            all_codes.append(codes[2][0][4*i].item()+128266+(2*4096))
            all_codes.append(codes[2][0][(4*i)+1].item()+128266+(3*4096))
            all_codes.append(codes[1][0][(2*i)+1].item()+128266+(4*4096))
            all_codes.append(codes[2][0][(4*i)+2].item()+128266+(5*4096))
            all_codes.append(codes[2][0][(4*i)+3].item()+128266+(6*4096))
        return all_codes

    def redistribute_codes(code_list):
        layer_1 = []
        layer_2 = []
        layer_3 = []
        for i in range((len(code_list)+1)//7):
            layer_1.append(code_list[7*i])
            layer_2.append(code_list[7*i+1]-4096)
            layer_3.append(code_list[7*i+2]-(2*4096))
            layer_3.append(code_list[7*i+3]-(3*4096))
            layer_2.append(code_list[7*i+4]-(4*4096))
            layer_3.append(code_list[7*i+5]-(5*4096))
            layer_3.append(code_list[7*i+6]-(6*4096))
        codes = [torch.tensor(layer_1).unsqueeze(0).cuda(),
                torch.tensor(layer_2).unsqueeze(0).cuda(),
                torch.tensor(layer_3).unsqueeze(0).cuda()]
        audio_hat = snac_model.decode(codes)
        return audio_hat

    for r in tqdm(rows):
        clone_from_audio = os.path.join("common-voice", r['language'], 'audio', r['audio_filename'])
        filename = os.path.join(r['output'], f"{r['index']}-{r['retry']}.mp3")

        try:
            sf.read(filename)
            continue
        except:
            pass

        try:
            chosen_voice = 'tara'
            prompts = [r['target_text']]

            all_input_ids = []
            for prompt in prompts:
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                all_input_ids.append(input_ids)

            start_token = torch.tensor([[ 128259]], dtype=torch.int64) # Start of human
            end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64) # End of text, End of human

            all_modified_input_ids = []
            for input_ids in all_input_ids:
                modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1) # SOH SOT Text EOT EOH
                all_modified_input_ids.append(modified_input_ids)

            all_padded_tensors = []
            all_attention_masks = []
            max_length = max([modified_input_ids.shape[1] for modified_input_ids in all_modified_input_ids])
            for modified_input_ids in all_modified_input_ids:
                padding = max_length - modified_input_ids.shape[1]
                padded_tensor = torch.cat([torch.full((1, padding), 128263, dtype=torch.int64), modified_input_ids], dim=1)
                attention_mask = torch.cat([torch.zeros((1, padding), dtype=torch.int64), torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64)], dim=1)
                all_padded_tensors.append(padded_tensor)
                all_attention_masks.append(attention_mask)

            all_padded_tensors = torch.cat(all_padded_tensors, dim=0)
            all_attention_masks = torch.cat(all_attention_masks, dim=0)

            input_ids = all_padded_tensors.to("cuda")
            attention_mask = all_attention_masks.to("cuda")

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1200,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    num_return_sequences=1,
                    eos_token_id=128258,
                )

            token_to_find = 128257
            token_to_remove = 128258

            # Check if the token exists in the tensor
            token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

            if len(token_indices[1]) > 0:
                last_occurrence_idx = token_indices[1][-1].item()
                cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
            else:
                cropped_tensor = generated_ids

            mask = cropped_tensor != token_to_remove
            processed_rows = []
            for row in cropped_tensor:
                # Apply the mask to each row
                masked_row = row[row != token_to_remove]
                processed_rows.append(masked_row)

            code_lists = []
            for row in processed_rows:
                # row is a 1D tensor with its own length
                row_length = row.size(0)
                new_length = (row_length // 7) * 7  # largest multiple of 7 that fits in this row
                trimmed_row = row[:new_length]
                trimmed_row = [t - 128266 for t in trimmed_row]
                code_lists.append(trimmed_row)

            samples = redistribute_codes(code_lists[0])
            audio_waveform = samples.detach().squeeze().to("cpu").numpy()

            os.makedirs(os.path.split(filename)[0], exist_ok = True)
            sf.write(filename, audio_waveform, 24000)
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
