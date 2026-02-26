import click
import requests
import os
import json
import pandas as pd
import re
from collections import defaultdict
from functools import partial
from multiprocess import Pool
from tqdm import tqdm

API_KEY = os.environ.get('API_KEY')
BASE_URL = os.environ.get('BASE_URL', "https://api.deepinfra.com/v1/openai")
MODEL_NAME = os.environ.get('MODEL_NAME', "Qwen/Qwen2.5-72B-Instruct")

_FEW_SHOT_USER = """
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

_FEW_SHOT_ASSISTANT = """
The audio features a teenage male with a North American accent, speaking in a neutral tone at a moderate pace. The speech is clear and fluent throughout, with a very low pitch that adds a distinctive quality to his voice. The environment is confined, giving the audio a somewhat boxed-in sound. The speech is notably monotone, lacking variation in pitch and emotion, which might make it sound a bit flat or unengaging to the listener. Despite the clear delivery, the overall audio quality is poor, with noticeable background noise that can distract from the message. Audio generated could be categorized as a casual informational segment, perhaps similar to a brief tutorial or informational clip, rather than a formal presentation or audiobook.
""".strip()

_FEW_SHOT_ASSISTANT_CATEGORY = """
Speaker profile: Teenage male with a North American accent.
Vocal qualities: Very low pitch, steady breath control, and clear articulation. The voice should sound grounded and slightly deep for his age.
Speaking style: Neutral and restrained. Maintain a monotone delivery with minimal pitch variation and limited emotional expression. Avoid dramatic emphasis or expressive inflection.
Pace: Moderate and consistent throughout. Do not rush or slow down noticeably.
Fluency: Smooth and continuous speech with no hesitations, filler words, or stutters.
Acoustic environment: Simulate a confined indoor space that produces a slightly boxed or enclosed sound.
Audio quality: Intentionally include mild background noise and subtle recording imperfections to reflect lower fidelity conditions. The noise should be noticeable but not overpower speech intelligibility.
Content style: Casual informational segment. The tone should resemble a short tutorial or brief explanatory clip rather than a formal presentation, narration, or audiobook.
""".strip()

FEW_SHOT_MESSAGES = [
    {'role': 'user', 'content': _FEW_SHOT_USER},
    {'role': 'assistant', 'content': _FEW_SHOT_ASSISTANT},
]

FEW_SHOT_MESSAGES_CATEGORY = [
    {'role': 'user', 'content': _FEW_SHOT_USER},
    {'role': 'assistant', 'content': _FEW_SHOT_ASSISTANT_CATEGORY},
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


cyrillic_characters = [
    # Basic Cyrillic Alphabet
    'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я',
    'а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я',

    # Extended Cyrillic Characters
    'Ѐ', 'Ђ', 'Ѓ', 'Є', 'Ѕ', 'І', 'Ї', 'Ј', 'Љ', 'Њ', 'Ћ', 'Ќ', 'Ѝ', 'Ў', 'Џ', 'Ѡ', 'Ѣ', 'Ѥ', 'Ѧ', 'Ѩ', 'Ѫ', 'Ѭ', 'Ѯ', 'Ѱ', 'Ѳ', 'Ѵ', 'Ҁ', 'Ҋ', 'Ҍ', 'Ҏ', 'Ґ', 'Ғ', 'Ҕ', 'Җ', 'Ҙ', 'Қ', 'Ҝ', 'Ҟ', 'Ҡ', 'Ң', 'Ҥ', 'Ҧ', 'Ҩ', 'Ҫ', 'Ҭ', 'Ү', 'Ұ', 'Ҳ', 'Ҵ', 'Ҷ', 'Ҹ', 'Һ', 'Ҽ', 'Ҿ', 'Ӏ', 'Ӂ', 'Ӄ', 'Ӆ', 'Ӈ', 'Ӊ', 'Ӌ', 'Ӎ', 'Ӑ', 'Ӓ', 'Ӕ', 'Ӗ', 'Ә', 'Ӛ', 'Ӝ', 'Ӟ', 'Ӡ', 'Ӣ', 'Ӥ', 'Ӧ', 'Ө', 'Ӫ', 'Ӭ', ' Ӯ', 'Ӱ', 'Ӳ', 'Ӵ', 'Ӷ', 'Ӹ', 'Ӻ', 'Ӽ', 'Ӿ', 'ӿ', 'Ԁ', 'Ԃ', 'Ԅ', 'Ԇ', 'Ԉ', 'Ԋ', 'Ԍ', 'Ԏ', 'Ԑ', 'Ԓ', 'Ԕ', 'Ԗ', 'Ԙ', 'Ԛ', 'Ԝ', 'Ԟ', 'Ԡ', 'Ԣ', 'ԥ', 'Ԧ', 'Ԩ', 'Ԫ', 'Ԭ', 'Ԯ', '԰', 'Բ', 'Դ', 'Զ', 'Ը', 'Ժ', 'Լ', 'Ծ',
    'ѐ', 'ђ', 'ѓ', 'є', 'ѕ', 'і', 'ї', 'ј', 'љ', 'њ', 'ћ', 'ќ', 'ѝ', 'ў', 'џ', 'ѡ', 'ѣ', 'ѥ', 'ѧ', 'ѩ', 'ѫ', 'ѭ', 'ѯ', 'ѱ', 'ѳ', 'ѵ', 'ҁ', 'ҋ', 'ҍ', 'ҏ', 'ґ', 'ғ', 'ҕ', 'җ', 'ҙ', 'қ', 'ҝ', 'ҟ', 'ҡ', 'ң', 'ҥ', 'ҧ', 'ҵ', 'ҫ', 'ҭ', 'ү', 'ұ', 'ҳ', 'ҵ', 'җ', 'ҹ', 'һ', 'ҽ', 'ҿ', 'ӏ', 'ӂ', 'ӄ', 'ӆ', 'ӈ', 'ӊ', 'ӌ', 'ӎ', 'ạ', 'ӓ', 'ӕ', 'ӗ', 'ә', 'ӛ', 'ӝ', 'ӟ', 'ӡ', 'ӣ', 'ӥ', 'ӧ', 'ө', 'ӫ', 'ӭ', 'ӯ', 'ӱ', 'ӳ', 'ӵ', 'ғ', 'ӷ', 'ӹ', 'ӻ', 'ӽ', 'ӿ', 'ԁ', 'ԃ', 'ԅ', 'ԇ', 'ԉ', 'ԋ', 'ԍ', 'ԏ', 'ԑ', 'ԓ', 'ԕ', 'ԗ', 'ԙ', 'ԛ', 'ԝ', 'ԟ', 'ԡ', 'ԣ', 'ԥ', 'ԧ', 'ԩ', 'ԫ', 'ԭ', 'ԯ', 'Ա', 'Գ', 'Ե', 'Է', 'Թ', 'Ի', 'Խ', 'Կ'
]

cyrillic_characters = set(cyrillic_characters)

weird_chars = {
 '\x81',
 '\x8a',
 '\x8b',
 '\x8c',
 '\x8d',
 '\x8f',
 '\x90',
 '\x96',
 '\x9d',
 '\x9f',
 '¡',
 '¤',
 '¥',
 '§',
 '¨',
 'ª',
 '«',
 '¬',
 '\xad',
 '¯',
 '°',
 '³',
 '¶',
 '·',
 '¸',
 '¹',
 'º',
 '»',
 '¼',
 '½',
 '¾',
 'ã',
 'ä',
 'å',
 'æ',
 'ç',
 'è',
 'é',
 'ï',
 'Œ',
 'œ',
 'Š',
 'š',
 'Ÿ',
 'ˆ',
 '˜',
 '–',
 '�',
 '\u2018',
 '‚',
 '„',
 '€'}


def detect_indian(text):
    for ch in text:
        code_point = ord(ch)
        if (
            0x0900 <= code_point <= 0x097F or  # Devanagari
            0x0980 <= code_point <= 0x09FF or  # Bengali
            0x0A00 <= code_point <= 0x0A7F or  # Gurmukhi
            0x0A80 <= code_point <= 0x0AFF or  # Gujarati
            0x0B00 <= code_point <= 0x0B7F or  # Oriya
            0x0B80 <= code_point <= 0x0BFF or  # Tamil
            0x0C00 <= code_point <= 0x0C7F or  # Telugu
            0x0C80 <= code_point <= 0x0CFF or  # Kannada
            0x0D00 <= code_point <= 0x0D7F     # Malayalam
        ):
            return True
    return False


def detect_mandarin(text):
    for char in text:
        codepoint = ord(char)
        if (
            0x4E00 <= codepoint <= 0x9FFF or   # CJK Unified Ideographs
            0x3400 <= codepoint <= 0x4DBF or   # CJK Unified Ideographs Extension A
            0x20000 <= codepoint <= 0x2A6DF or # Extension B
            0x2A700 <= codepoint <= 0x2B73F or # Extension C
            0x2B740 <= codepoint <= 0x2B81F or # Extension D
            0x2B820 <= codepoint <= 0x2CEAF or # Extension E
            0x2CEB0 <= codepoint <= 0x2EBEF    # Extension F
        ):
            return True
    return False


def detect_russian(text):
    russian_pattern = re.compile(r'[\u0400-\u04FF]+')
    return bool(russian_pattern.search(text))


def detect_arabic(text):
    arabic_pattern = re.compile(r'[\u0600-\u06FF]+')
    return bool(arabic_pattern.search(text))


def detect_ngram_repetitions(text, n=10, word=True):
    tokens = text.split() if word else text
    ngrams = defaultdict(int)
    for i in range(len(tokens) - n + 1):
        ngram = ' '.join(tokens[i:i+n])
        ngrams[ngram] += 1
    return {ngram: count for ngram, count in ngrams.items() if count > 1}


def accept(d, min_len=20):
    d = d.strip()

    if '**' in d:
        return False

    if len(d.split()) < min_len:
        return False

    if len(set(d) & cyrillic_characters):
        return False

    if len(set(d) & weird_chars):
        return False

    repeated = detect_ngram_repetitions(d, n=5)
    for v in repeated.values():
        if v > 3:
            return False

    if detect_arabic(d):
        return False

    if detect_russian(d):
        return False

    if detect_indian(d):
        return False

    if detect_mandarin(d):
        return False

    return True


def build_fluency_label(fluency_data):
    fluency = json.loads(fluency_data)
    parts = []
    for i in range(len(fluency['timestamp'])):
        start, end = fluency['timestamp'][i]
        label = fluency['label'][i]
        parts.append(f'{start:02d}:{end:02d}:{label}')
    return ', '.join(parts)


def build_prompt(r):
    fluency_label = build_fluency_label(r['fluency'])
    return f"""
based on audio attributes,
fluency timestamp: {fluency_label}
accent: {r['accent']}
gender: {r['gender']}
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


def call_api(messages, headers, url, min_len, max_len, min_newlines=None, max_newlines=None, max_retries=5):
    for _ in range(max_retries):
        response = requests.post(url, headers=headers, json={
            'model': MODEL_NAME,
            'messages': messages,
        })
        content = response.json()['choices'][0]['message']['content']
        if not (min_len <= len(content) <= max_len):
            continue
        if min_newlines is not None and max_newlines is not None:
            newline_count = content.count('\n')
            if not (min_newlines <= newline_count <= max_newlines):
                continue
        if not accept(content):
            continue
        return content
    return None


def process_batch(indices_device_pair, folder):
    rows, device = indices_device_pair

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}',
    }
    url = f'{BASE_URL}/chat/completions'

    for r in tqdm(rows):
        filename = os.path.join(folder, str(r['index']) + '.json')
        try:
            with open(filename) as fopen:
                json.load(fopen)
                continue
        except Exception:
            pass

        prompt = build_prompt(r)

        description_category = call_api(
            messages=FEW_SHOT_MESSAGES_CATEGORY + [{'role': 'user', 'content': prompt}],
            headers=headers,
            url=url,
            min_len=800,
            max_len=2000,
            min_newlines=5,
            max_newlines=15,
        )
        if description_category is None:
            continue

        description = call_api(
            messages=FEW_SHOT_MESSAGES + [{'role': 'user', 'content': prompt}],
            headers=headers,
            url=url,
            min_len=600,
            max_len=1300,
        )
        if description is None:
            continue

        os.makedirs(os.path.split(filename)[0], exist_ok=True)
        for c in ('quality', 'phonemes'):
            r[c] = r[c].tolist()
        with open(filename, 'w') as fopen:
            json.dump({**r, 'description': description, 'description_category': description_category}, fopen)


@click.command()
@click.option('--file')
@click.option('--replication', default=5)
@click.option('--folder', default='output')
def main(file, replication, folder):

    devices = list(range(replication))
    df = pd.read_parquet(file)
    df['index'] = df.index
    rows = df.to_dict(orient='records')

    pending = []
    for r in tqdm(rows):
        filename = os.path.join(folder, str(r['index']) + '.json')
        if os.path.exists(filename):
            try:
                with open(filename) as fopen:
                    json.load(fopen)
                continue
            except Exception:
                pass
        pending.append(r)

    if pending:
        batches = list(chunks(pending, devices))
        loop_partial = partial(process_batch, folder=folder)
        with Pool(len(devices)) as pool:
            pool.map(loop_partial, batches)


if __name__ == '__main__':
    main()
