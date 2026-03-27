# TTS Evaluation

Benchmarking multilingual TTS models across 76 languages using Character Error Rate (CER) and MOS based on UTMOSv2.

![Benchmark scatter](scatter_results.png)

## Models

| Model | Description |
|-------|-------------|
| **Dia TTS** | [Nari Labs Dia TTS](https://github.com/nari-labs/dia) |
| **Multilingual TTS 0.6B** | [Scicom-intl/Multilingual-Expressive-TTS-0.6B](https://huggingface.co/Scicom-intl/Multilingual-Expressive-TTS-0.6B) |
| **Multilingual TTS 1.7B** | [Scicom-intl/Multilingual-Expressive-TTS-1.7B](https://huggingface.co/Scicom-intl/Multilingual-Expressive-TTS-1.7B) |
| **Orpheus** | [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) |
| **Chatterbox** | [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) — 23 languages only |
| **Fish Audio S2 Pro** | [Fish Audio S2 Pro](https://github.com/fishaudio/fish-speech) |
| **Qwen3 TTS** | [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) — 11 languages only |

## Setup

```bash
pip install -r requirements.txt
```

## Run Generations

Each prompt is generated **twice** and the scores are averaged to reduce variance. We also upload all the generations done by us at [Scicom-intl/Evaluation-Multilingual-VC](https://huggingface.co/datasets/Scicom-intl/Evaluation-Multilingual-VC)

### Dia TTS

```bash
python3 dia_tts.py --output 'dia-tts'
```

### Scicom Multilingual TTS

```bash
# 0.6B
MODEL_NAME="Scicom-intl/Multilingual-Expressive-TTS-0.6B" python3 multilingual_tts.py \
  --speaker 'multilingual-tts_audio_Grace' --output 'multilingual-tts-0.6b'

# 1.7B
MODEL_NAME="Scicom-intl/Multilingual-Expressive-TTS-1.7B" python3 multilingual_tts.py \
  --speaker 'multilingual-tts_audio_Grace' --output 'multilingual-tts-1.7b'
```

### Orpheus

```bash
python3 orpheus.py --output 'orpheus'
```

### Chatterbox

```bash
python3 chatterbox.py --output 'chatterbox'
```

### Fish Audio S2 Pro

```bash
python3 fishspeech2.py --output 'fishspeech2'
```

### Qwen3 TTS

```bash
python3 qwen3_tts.py --output 'qwen3_tts'
```

## Evaluate

### CER

```bash
python3 calculate_cer.py --output_folder "dia-tts"               --output "dia-tts-cer"
python3 calculate_cer.py --output_folder "multilingual-tts-0.6b" --output "multilingual-tts-0.6b-cer"
python3 calculate_cer.py --output_folder "multilingual-tts-1.7b" --output "multilingual-tts-1.7b-cer"
python3 calculate_cer.py --output_folder "orpheus"               --output "orpheus-cer"
python3 calculate_cer.py --output_folder "chatterbox"            --output "chatterbox-cer"
python3 calculate_cer.py --output_folder "fishspeech2"           --output "fishspeech2-cer"
python3 calculate_cer.py --output_folder "qwen3_tts"             --output "qwen3_tts-cer"
```

### MOS

We evaluate using [Scicom-AI-Enterprise-Organization/faster-UTMOSv2](https://github.com/Scicom-AI-Enterprise-Organization/faster-UTMOSv2) with 5 repetitions.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 calculate_mos.py --output_folder "dia-tts"  --output "dia-tts-mos"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 calculate_mos.py --output_folder "multilingual-tts-0.6b" --output "multilingual-tts-0.6b-mos" --replication 5
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 calculate_mos.py --output_folder "multilingual-tts-1.7b" --output "multilingual-tts-1.7b-mos" --replication 5
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 calculate_mos.py --output_folder "orpheus" --output "orpheus-mos" --replication 5
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 calculate_mos.py --output_folder "chatterbox" --output "chatterbox-mos" --replication 5
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 calculate_mos.py --output_folder "fishspeech2" --output "fishspeech2-mos" --replication 5
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 calculate_mos.py --output_folder "qwen3_tts" --output "qwen3_tts-mos" --replication 5
```

#### Dia TTS

```
af: 2.2471 (131 samples)
am: 1.7337 (252 samples)
ar: 1.7033 (496 samples)
as: 1.7492 (379 samples)
az: 1.9577 (95 samples)
ba: 1.7044 (498 samples)
be: 1.7727 (500 samples)
bg: 1.5621 (500 samples)
bn: 1.8448 (500 samples)
br: 1.6231 (499 samples)
ca: 2.0167 (500 samples)
cs: 1.9180 (499 samples)
cy: 1.9149 (500 samples)
da: 1.9296 (499 samples)
de: 2.1887 (499 samples)
el: 1.7758 (500 samples)
en: 1.9760 (499 samples)
es: 2.1636 (497 samples)
et: 2.4030 (500 samples)
eu: 2.3006 (500 samples)
fa: 1.6383 (500 samples)
fi: 1.8968 (500 samples)
fr: 2.0537 (500 samples)
gl: 2.0420 (500 samples)
ha: 1.8402 (500 samples)
he: 1.7136 (392 samples)
hi: 1.6255 (500 samples)
ht: 1.9812 (5 samples)
hu: 2.0455 (500 samples)
hy-AM: 1.7129 (500 samples)
id: 2.0662 (500 samples)
is: 2.0062 (9 samples)
it: 2.2322 (500 samples)
ja: 1.8723 (456 samples)
ka: 1.8551 (500 samples)
kk: 1.6974 (500 samples)
ko: 1.5789 (465 samples)
lo: 1.8653 (26 samples)
lt: 2.1388 (500 samples)
lv: 1.8705 (500 samples)
mk: 1.5074 (500 samples)
ml: 1.8631 (495 samples)
mn: 1.5779 (500 samples)
mr: 1.7815 (500 samples)
mt: 2.0890 (500 samples)
ne-NP: 1.7211 (287 samples)
nl: 2.1065 (500 samples)
nn-NO: 1.8905 (423 samples)
oc: 1.8265 (274 samples)
pa-IN: 1.7590 (500 samples)
pl: 1.8804 (500 samples)
ps: 1.5740 (500 samples)
pt: 1.9530 (498 samples)
ro: 1.9428 (500 samples)
ru: 1.6516 (499 samples)
sd: 1.6618 (40 samples)
sk: 1.8710 (495 samples)
sl: 1.7868 (500 samples)
sq: 1.9454 (500 samples)
sr: 1.5622 (500 samples)
sv-SE: 1.8089 (500 samples)
sw: 2.2197 (500 samples)
ta: 1.8343 (500 samples)
te: 1.8003 (66 samples)
tg: 1.4999 (69 samples)
th: 1.8742 (498 samples)
tk: 1.9589 (499 samples)
tr: 1.8553 (500 samples)
tt: 1.7529 (500 samples)
uk: 1.6260 (500 samples)
ur: 1.5768 (500 samples)
uz: 2.1818 (500 samples)
vi: 1.5317 (498 samples)
yi: 1.7104 (222 samples)
yo: 1.7179 (500 samples)
zh-CN: 1.8971 (488 samples)
zh-HK: 1.8981 (371 samples)
zh-TW: 1.9719 (466 samples)

Global average: 1.8575
```

#### 0.6B

```
af: 3.3333 (131 samples)
am: 3.2761 (252 samples)
ar: 3.1850 (496 samples)
as: 3.2322 (379 samples)
az: 3.1916 (95 samples)
ba: 3.0963 (498 samples)
be: 3.2335 (500 samples)
bg: 3.2341 (500 samples)
bn: 3.2044 (500 samples)
br: 3.1423 (499 samples)
ca: 3.2201 (500 samples)
cs: 3.2258 (499 samples)
cy: 3.2941 (500 samples)
da: 3.3018 (499 samples)
de: 3.2963 (499 samples)
el: 3.2046 (500 samples)
en: 3.5712 (499 samples)
es: 3.2050 (497 samples)
et: 3.2800 (500 samples)
eu: 3.2379 (500 samples)
fa: 3.2019 (500 samples)
fi: 3.2996 (500 samples)
fr: 3.1638 (500 samples)
gl: 3.2531 (500 samples)
ha: 3.3168 (500 samples)
he: 3.2590 (392 samples)
hi: 3.1946 (500 samples)
ht: 3.2869 (5 samples)
hu: 3.2493 (500 samples)
hy-AM: 3.1522 (500 samples)
id: 3.2205 (500 samples)
is: 3.2593 (9 samples)
it: 3.2260 (500 samples)
ja: 3.1508 (456 samples)
ka: 3.1751 (500 samples)
kk: 3.1333 (500 samples)
ko: 3.0851 (465 samples)
lo: 3.2109 (26 samples)
lt: 3.2714 (500 samples)
lv: 3.2486 (500 samples)
mk: 3.2366 (500 samples)
ml: 3.1769 (495 samples)
mn: 3.1648 (500 samples)
mr: 3.1820 (500 samples)
mt: 3.2862 (500 samples)
ne-NP: 3.2393 (287 samples)
nl: 3.3584 (500 samples)
nn-NO: 3.3027 (423 samples)
oc: 3.2293 (274 samples)
pa-IN: 3.2285 (500 samples)
pl: 3.2618 (500 samples)
ps: 3.2188 (500 samples)
pt: 3.2072 (498 samples)
ro: 3.2615 (500 samples)
ru: 3.1882 (499 samples)
sd: 3.2792 (40 samples)
sk: 3.1526 (495 samples)
sl: 3.2194 (500 samples)
sq: 3.2027 (500 samples)
sr: 3.0163 (500 samples)
sv-SE: 3.2189 (500 samples)
sw: 3.3415 (500 samples)
ta: 3.1910 (500 samples)
te: 3.2538 (66 samples)
tg: 3.1549 (69 samples)
th: 3.2146 (498 samples)
tk: 3.2020 (499 samples)
tr: 3.1076 (500 samples)
tt: 3.1428 (500 samples)
uk: 3.2035 (500 samples)
ur: 3.2268 (500 samples)
uz: 3.2229 (500 samples)
vi: 3.2520 (498 samples)
yi: 3.2762 (222 samples)
yo: 3.2922 (500 samples)
zh-CN: 3.2257 (488 samples)
zh-HK: 3.2032 (371 samples)
zh-TW: 3.2620 (466 samples)

Global average: 3.2273
```

#### 1.7B

```
af: 3.3340 (131 samples)
am: 3.2847 (252 samples)
ar: 3.1970 (496 samples)
as: 3.2721 (379 samples)
az: 3.2355 (95 samples)
ba: 3.0950 (498 samples)
be: 3.2159 (500 samples)
bg: 3.2276 (500 samples)
bn: 3.2457 (500 samples)
br: 3.1386 (499 samples)
ca: 3.2539 (500 samples)
cs: 3.2480 (499 samples)
cy: 3.3059 (500 samples)
da: 3.3080 (499 samples)
de: 3.2864 (499 samples)
el: 3.2021 (500 samples)
en: 3.5550 (499 samples)
es: 3.1922 (497 samples)
et: 3.3047 (500 samples)
eu: 3.2556 (500 samples)
fa: 3.2031 (500 samples)
fi: 3.3064 (500 samples)
fr: 3.1835 (500 samples)
gl: 3.2519 (500 samples)
ha: 3.3397 (500 samples)
he: 3.2859 (392 samples)
hi: 3.1871 (500 samples)
ht: 3.1869 (5 samples)
hu: 3.2711 (500 samples)
hy-AM: 3.1496 (500 samples)
id: 3.2195 (500 samples)
is: 3.3052 (9 samples)
it: 3.2313 (500 samples)
ja: 3.1533 (456 samples)
ka: 3.2063 (500 samples)
kk: 3.1151 (500 samples)
ko: 3.1004 (465 samples)
lo: 3.2337 (26 samples)
lt: 3.2864 (500 samples)
lv: 3.2288 (500 samples)
mk: 3.2285 (500 samples)
ml: 3.1492 (495 samples)
mn: 3.1630 (500 samples)
mr: 3.1915 (500 samples)
mt: 3.3114 (500 samples)
ne-NP: 3.2047 (287 samples)
nl: 3.3422 (500 samples)
nn-NO: 3.3147 (423 samples)
oc: 3.2672 (274 samples)
pa-IN: 3.2304 (500 samples)
pl: 3.2262 (500 samples)
ps: 3.2384 (500 samples)
pt: 3.1941 (498 samples)
ro: 3.2850 (500 samples)
ru: 3.1888 (499 samples)
sd: 3.2637 (40 samples)
sk: 3.1466 (495 samples)
sl: 3.2096 (500 samples)
sq: 3.2484 (500 samples)
sr: 2.9958 (500 samples)
sv-SE: 3.2233 (500 samples)
sw: 3.3559 (500 samples)
ta: 3.2191 (500 samples)
te: 3.2042 (66 samples)
tg: 3.1391 (69 samples)
th: 3.2309 (498 samples)
tk: 3.2263 (499 samples)
tr: 3.1340 (500 samples)
tt: 3.1505 (500 samples)
uk: 3.1967 (500 samples)
ur: 3.2193 (500 samples)
uz: 3.2426 (500 samples)
vi: 3.2658 (498 samples)
yi: 3.3105 (222 samples)
yo: 3.3649 (500 samples)
zh-CN: 3.2459 (488 samples)
zh-HK: 3.2052 (371 samples)
zh-TW: 3.2347 (466 samples)

Global average: 3.2330
```

#### Chatterbox

```
ar: 2.7880 (496 samples)
da: 2.8204 (499 samples)
de: 3.0291 (499 samples)
el: 2.7216 (500 samples)
en: 3.2585 (499 samples)
es: 2.8846 (497 samples)
fi: 2.7206 (500 samples)
fr: 2.9151 (500 samples)
he: 2.8807 (392 samples)
hi: 2.8223 (500 samples)
it: 2.8648 (500 samples)
ja: 2.7015 (456 samples)
ko: 2.8574 (465 samples)
nl: 3.0348 (500 samples)
nn-NO: 2.8826 (423 samples)
pl: 2.7936 (500 samples)
pt: 2.8441 (498 samples)
ru: 2.8646 (499 samples)
sv-SE: 2.6231 (500 samples)
sw: 2.9911 (500 samples)
tr: 2.7691 (500 samples)
zh-CN: 2.6391 (488 samples)
zh-TW: 2.6253 (466 samples)

Global average: 2.8405
```

#### FishSpeech S2

```
af: 3.0865 (131 samples)
am: 2.7832 (252 samples)
ar: 2.9378 (496 samples)
as: 2.9876 (379 samples)
az: 3.0641 (95 samples)
ba: 2.7391 (498 samples)
be: 3.0450 (500 samples)
bg: 2.9889 (500 samples)
bn: 3.0289 (500 samples)
br: 2.8960 (499 samples)
ca: 3.0211 (500 samples)
cs: 3.0426 (499 samples)
cy: 3.0701 (500 samples)
da: 3.0309 (499 samples)
de: 3.2279 (499 samples)
el: 2.9599 (500 samples)
en: 3.4690 (499 samples)
es: 3.0896 (497 samples)
et: 3.1152 (500 samples)
eu: 3.0561 (500 samples)
fa: 2.9902 (500 samples)
fi: 3.1364 (500 samples)
fr: 3.1376 (500 samples)
gl: 3.1084 (500 samples)
ha: 2.9785 (500 samples)
he: 3.0091 (392 samples)
hi: 3.0060 (500 samples)
ht: 2.8443 (5 samples)
hu: 3.1362 (500 samples)
hy-AM: 2.9400 (500 samples)
id: 3.0981 (500 samples)
is: 3.1808 (9 samples)
it: 3.0660 (500 samples)
ja: 2.7074 (456 samples)
ka: 3.0342 (500 samples)
kk: 2.8745 (500 samples)
ko: 2.9186 (465 samples)
lo: 2.6337 (26 samples)
lt: 3.1242 (500 samples)
lv: 3.0345 (500 samples)
mk: 3.0280 (500 samples)
ml: 2.5700 (495 samples)
mn: 2.9345 (500 samples)
mr: 2.8192 (500 samples)
mt: 2.9729 (500 samples)
ne-NP: 2.7929 (287 samples)
nl: 3.1947 (500 samples)
nn-NO: 3.0548 (423 samples)
oc: 2.9600 (274 samples)
pa-IN: 2.5228 (500 samples)
pl: 3.0836 (500 samples)
ps: 2.8921 (500 samples)
pt: 3.0262 (498 samples)
ro: 3.0390 (500 samples)
ru: 3.0412 (499 samples)
sd: 2.8330 (40 samples)
sk: 2.9298 (495 samples)
sl: 2.9982 (500 samples)
sq: 2.9999 (500 samples)
sr: 2.8833 (500 samples)
sv-SE: 2.9870 (500 samples)
sw: 2.9219 (500 samples)
ta: 2.7453 (500 samples)
te: 2.5483 (66 samples)
tg: 2.9765 (69 samples)
th: 2.8463 (498 samples)
tk: 2.9744 (499 samples)
tr: 2.8964 (500 samples)
tt: 2.8753 (500 samples)
uk: 2.9634 (500 samples)
ur: 3.0278 (500 samples)
uz: 3.0313 (500 samples)
vi: 2.8471 (498 samples)
yi: 2.9735 (222 samples)
yo: 2.8477 (500 samples)
zh-CN: 3.0697 (488 samples)
zh-HK: 2.9997 (371 samples)
zh-TW: 2.9403 (466 samples)

Global average: 2.9698
```

#### Qwen3 TTS

```
de: 2.7600 (499 samples)
en: 3.0221 (499 samples)
es: 2.6931 (497 samples)
fr: 2.5720 (500 samples)
it: 2.4546 (500 samples)
ja: 2.5274 (456 samples)
ko: 2.4355 (465 samples)
pt: 2.5732 (498 samples)
ru: 2.3231 (499 samples)
zh-CN: 2.7834 (488 samples)
zh-TW: 2.5358 (466 samples)

Global average: 2.6073
```

## Results

Summary across all evaluated languages. Full per-language heatmap below.

| Model | Languages | CER ↓ |
|-------|:---------:|:-----:|
| Dia TTS | 76 | 0.8131 |
| Multilingual TTS 0.6B | 76 | 0.2384 |
| Multilingual TTS 1.7B | 76 | **0.2362** |
| Orpheus | 76 | 0.6075 |
| Chatterbox | 23 | 0.1698 |
| Fish Audio S2 Pro | 76 | 0.2370 |
| Qwen3 TTS | 11 | **0.1064** |

> Chatterbox covers 23 languages only; Qwen3 TTS covers 11 languages only. Their averages are not directly comparable to 76-language models.

### Full Breakdown (76 languages)

![Benchmark heatmap](benchmark_results.png)
