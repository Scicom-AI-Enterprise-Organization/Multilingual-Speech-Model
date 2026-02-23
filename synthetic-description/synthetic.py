import click
import requests
import os
import json
import requests
from functools import partial
from multiprocess import Pool
from tqdm import tqdm

API_KEY = os.environ.get('API_KEY')
BASE_URL = os.environ.get('BASE_URL', "https://api.deepinfra.com/v1/openai")
MODEL_NAME = os.environ.get('MODEL_NAME', "Qwen/Qwen2.5-72B-Instruct")

user = """
based on audio attributes,
fluency timestamp: 00:03:Fluent, 01:04:Fluent, 02:05:Fluent, 03:06:Fluent, 04:07:Fluent, 05:08:Fluent
accent: North America
gender: Male
age: 17
emotion: 中立/neutral
speed: moderate speed
noise: quite clear
reverberation: very confined sounding
speech monotony: quite monotone
SDR: extremely noisy
PESQ: bad speech quality
pitch: very low pitch
transcript: You can find information about a celebrity by searching their name online using a search engine like Google, or by checking their official social media accounts like Twitter, Instagram or Facebook.

generate expressive description for the audio, 
1. do not include transcript in the description, or mentioned based on content
2. do not use numbering for the age, use like middle-age, teenager, etc
3. based on transcript, try to conclude the category in the description, like audiobook, news, etc
4. do not mention metrics like PESQ and SDR, describe the metrics for non technical person
""".strip()

assistant = """
The audio features a teenage male with a North American accent, speaking in a neutral tone at a moderate pace. The speech is clear and fluent throughout, with a very low pitch that adds a distinctive quality to his voice. The environment is confined, giving the audio a somewhat boxed-in sound. The speech is notably monotone, lacking variation in pitch and emotion, which might make it sound a bit flat or unengaging to the listener. Despite the clear delivery, the overall audio quality is poor, with noticeable background noise that can distract from the message. Audio generated could be categorized as a casual informational segment, perhaps similar to a brief tutorial or informational clip, rather than a formal presentation or audiobook.
""".strip()

assistant_category = """
Speaker profile: Teenage male with a North American accent.
Vocal qualities: Very low pitch, steady breath control, and clear articulation. The voice should sound grounded and slightly deep for his age.
Speaking style: Neutral and restrained. Maintain a monotone delivery with minimal pitch variation and limited emotional expression. Avoid dramatic emphasis or expressive inflection.
Pace: Moderate and consistent throughout. Do not rush or slow down noticeably.
Fluency: Smooth and continuous speech with no hesitations, filler words, or stutters.
Acoustic environment: Simulate a confined indoor space that produces a slightly boxed or enclosed sound.
Audio quality: Intentionally include mild background noise and subtle recording imperfections to reflect lower fidelity conditions. The noise should be noticeable but not overpower speech intelligibility.
Content style: Casual informational segment. The tone should resemble a short tutorial or brief explanatory clip rather than a formal presentation, narration, or audiobook.
""".strip()

user_assistant = [
    {'role': 'user', 'content': user},
    {'role': 'assistant', 'content': assistant}
]

user_assistant_category = [
    {'role': 'user', 'content': user},
    {'role': 'assistant', 'content': assistant_category}
]

def chunks(l, devices):
    chunk_size = len(l) // len(devices)
    remainder = len(l) % len(devices)
    start = 0
    for i in range(len(devices)):
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        yield (l[start:end], devices[i])
        start = end

def loop(
    indices_device_pair,
    folder,
):
    rows, device = indices_device_pair

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}',
    }

    for r in tqdm(rows):
        filename = os.path.join(folder, r['index'] + '.json')
        try:
            with open(filename) as fopen:
                json.load(fopen)
                continue
        except:
            pass

        fluency = json.loads(row['fluency'])
        fluency_label = []
        for i in range(len(fluency['timestamp'])):
            start = fluency['timestamp'][i][0]
            end = fluency['timestamp'][i][1]
            l = fluency['label'][i]
            fluency_label.append(f'{start:02d}:{end:02d}:{l}')

        fluency_label = ', '.join(fluency_label)

        prompt = f"""
based on audio attributes,
fluency timestamp: {fluency_label}
accent: {r['accent']}
gender: {r['sex']}
age: {r['age']}
emotion: {r['emotion']}
speed: {r['speaking_rate_label']}
noise: {r['noise_label']}
reverberation: {r['reverberation_label']}
speech monotony: {r['speech_monotony_label']}
SDR: {r['sdr_label']}
PESQ: {r['pesq_label']}
pitch: {r['pitch_label']}
transcript: {r['text']}

generate expressive description for the audio, 
1. do not include transcript in the description
2. do not use numbering for the age, use like middle-age, teenager, etc
3. based on transcript, try to conclude the category in the description, like audiobook, news, etc
4. do not mention metrics like PESQ and SDR
""".strip()
        url = f'{BASE_URL}/chat/completions'
        json_data = {
            'model': MODEL_NAME,
            'messages': user_assistant_category[:] + [{"role": "user", "content": prompt}]
        }
        response = requests.post(url, headers=headers, json=json_data)
        category_response = response.json()['choices'][0]['message']['content']

        json_data = {
            'model': MODEL_NAME,
            'messages': user_assistant[:] + [{"role": "user", "content": prompt}]
        }
        response = requests.post(url, headers=headers, json=json_data)
        response = response.json()['choices'][0]['message']['content']
        
        os.makedirs(os.path.split(filename)[0], exist_ok = True)
        new_r = {**r, 'description': response, 'description_category': category_response}
        with open(filename, 'w') as fopen:
            json.dump(new_r, fopen)

@click.command()
@click.option('--file')
@click.option('--replication', default = 5)
@click.option('--folder', default = 'output')
def main(file, replication, folder):

    devices = list(range(replication))
    df = pd.read_parquet(file)
    df['index'] = df.index
    rows = df.to_dict(orient='records')
    filtered = []
    for r in tqdm(rows): 
        filename = os.path.join(folder, r['index'] + '.json')
        if os.path.exists(filename):
            try:
                with open(filename) as fopen:
                    json.load(fopen)
                continue
            except:
                pass
        filtered.append(r)

    if len(filtered):
        df_split = list(chunks(filtered, devices))

        loop_partial = partial(
            loop,
            folder=folder,
        )

        with Pool(len(devices)) as pool:
            pooled = pool.map(loop_partial, df_split)

if __name__ == '__main__':
    main()
