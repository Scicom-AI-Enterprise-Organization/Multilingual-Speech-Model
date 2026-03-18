# TTS Evaluation

## How to run generations

### Dia TTS

```bash
python3 dia_tts.py --output 'dia-tts'
```

### Scicom Multilingual TTS

#### 0.6B

```bash
MODEL_NAME="Scicom-intl/Multilingual-Expressive-TTS-0.6B" python3 multilingual_tts.py \
--speaker 'multilingual-tts_audio_Grace' --output 'multilingual-tts-0.6b'
```

#### 1.7B

```bash
MODEL_NAME="Scicom-intl/Multilingual-Expressive-TTS-1.7B" python3 multilingual_tts.py \
--speaker 'multilingual-tts_audio_Grace' --output 'multilingual-tts-1.7b'
```

### Orpheus

```bash
python3 orpheus.py --output 'orpheus'
```

### Chatterbox

### FishSpeech2

```bash
python3 fishspeech2.py --output 'fishspeech2'
```

## How to calculate CER

### Scicom Multilingual TTS

#### 0.6B

```bash
python3 calculate_cer.py --output_folder "multilingual-tts-0.6b" --output "multilingual-tts-0.6b-cer"
```

```
af: 0.1346 (131 samples)
am: 1.0000 (252 samples)
ar: 0.1417 (496 samples)
as: 0.9192 (379 samples)
az: 0.0416 (95 samples)
ba: 0.6995 (498 samples)
be: 0.0831 (500 samples)
bg: 0.1094 (500 samples)
bn: 0.3311 (500 samples)
br: 0.3311 (499 samples)
ca: 0.0740 (500 samples)
cs: 0.0651 (499 samples)
cy: 0.2840 (500 samples)
da: 0.1472 (499 samples)
de: 0.0197 (499 samples)
el: 0.1017 (500 samples)
en: 0.0200 (499 samples)
es: 0.0265 (497 samples)
et: 0.0972 (500 samples)
eu: 0.0817 (500 samples)
fa: 0.1138 (500 samples)
fi: 0.1024 (500 samples)
fr: 0.0511 (500 samples)
gl: 0.0698 (500 samples)
ha: 0.2233 (500 samples)
he: 0.2154 (392 samples)
hi: 0.0727 (500 samples)
ht: 0.1825 (5 samples)
hu: 0.0864 (500 samples)
hy-AM: 0.0886 (500 samples)
id: 0.0228 (500 samples)
is: 0.2057 (9 samples)
it: 0.0418 (500 samples)
ja: 0.3158 (456 samples)
ka: 0.0962 (500 samples)
kk: 0.1208 (500 samples)
ko: 0.0486 (465 samples)
lo: 0.9977 (26 samples)
lt: 0.0929 (500 samples)
lv: 0.1410 (500 samples)
mk: 0.0960 (500 samples)
ml: 0.9627 (495 samples)
mn: 0.1959 (500 samples)
mr: 0.1605 (500 samples)
mt: 0.2782 (500 samples)
ne-NP: 0.2093 (287 samples)
nl: 0.0338 (500 samples)
nn-NO: 0.1870 (423 samples)
oc: 0.2786 (274 samples)
pa-IN: 0.3422 (500 samples)
pl: 0.0906 (500 samples)
ps: 0.3355 (500 samples)
pt: 0.0806 (498 samples)
ro: 0.0449 (500 samples)
ru: 0.0388 (499 samples)
sd: 0.9323 (40 samples)
sk: 0.2118 (495 samples)
sl: 0.0951 (500 samples)
sq: 0.1794 (500 samples)
sr: 0.6302 (500 samples)
sv-SE: 0.0936 (500 samples)
sw: 0.1100 (500 samples)
ta: 0.0924 (500 samples)
te: 0.5294 (66 samples)
tg: 0.2685 (69 samples)
th: 0.1394 (498 samples)
tk: 0.4219 (499 samples)
tr: 0.0754 (500 samples)
tt: 0.2748 (500 samples)
uk: 0.0732 (500 samples)
ur: 0.0628 (500 samples)
uz: 0.2579 (500 samples)
vi: 0.5255 (498 samples)
yi: 0.4113 (222 samples)
yo: 0.5187 (500 samples)
zh-CN: 0.2820 (488 samples)
zh-HK: 0.6496 (371 samples)
zh-TW: 0.4257 (466 samples)

Global average: 0.2384
```

#### 1.7B

```bash
python3 calculate_cer.py --output_folder "multilingual-tts-1.7b" --output "multilingual-tts-1.7b-cer"
```

```
af: 0.1158 (131 samples)
am: 1.0000 (252 samples)
ar: 0.1384 (496 samples)
as: 0.9360 (379 samples)
az: 0.0342 (95 samples)
ba: 0.6757 (498 samples)
be: 0.0812 (500 samples)
bg: 0.1035 (500 samples)
bn: 0.3762 (500 samples)
br: 0.3344 (499 samples)
ca: 0.0751 (500 samples)
cs: 0.0728 (499 samples)
cy: 0.2776 (500 samples)
da: 0.1586 (499 samples)
de: 0.0186 (499 samples)
el: 0.1052 (500 samples)
en: 0.0230 (499 samples)
es: 0.0244 (497 samples)
et: 0.0892 (500 samples)
eu: 0.0797 (500 samples)
fa: 0.1260 (500 samples)
fi: 0.0989 (500 samples)
fr: 0.0517 (500 samples)
gl: 0.0688 (500 samples)
ha: 0.2299 (500 samples)
he: 0.2303 (392 samples)
hi: 0.0707 (500 samples)
ht: 0.1874 (5 samples)
hu: 0.0842 (500 samples)
hy-AM: 0.0948 (500 samples)
id: 0.0215 (500 samples)
is: 0.1868 (9 samples)
it: 0.0369 (500 samples)
ja: 0.3074 (456 samples)
ka: 0.1074 (500 samples)
kk: 0.1019 (500 samples)
ko: 0.0483 (465 samples)
lo: 0.9958 (26 samples)
lt: 0.0934 (500 samples)
lv: 0.1239 (500 samples)
mk: 0.0906 (500 samples)
ml: 0.9625 (495 samples)
mn: 0.1934 (500 samples)
mr: 0.1586 (500 samples)
mt: 0.2769 (500 samples)
ne-NP: 0.2154 (287 samples)
nl: 0.0216 (500 samples)
nn-NO: 0.1951 (423 samples)
oc: 0.2707 (274 samples)
pa-IN: 0.3409 (500 samples)
pl: 0.0744 (500 samples)
ps: 0.3385 (500 samples)
pt: 0.0798 (498 samples)
ro: 0.0423 (500 samples)
ru: 0.0381 (499 samples)
sd: 0.9308 (40 samples)
sk: 0.2215 (495 samples)
sl: 0.1029 (500 samples)
sq: 0.1697 (500 samples)
sr: 0.6588 (500 samples)
sv-SE: 0.1044 (500 samples)
sw: 0.1055 (500 samples)
ta: 0.0887 (500 samples)
te: 0.5025 (66 samples)
tg: 0.2542 (69 samples)
th: 0.1681 (498 samples)
tk: 0.3701 (499 samples)
tr: 0.0619 (500 samples)
tt: 0.2657 (500 samples)
uk: 0.0722 (500 samples)
ur: 0.0654 (500 samples)
uz: 0.2629 (500 samples)
vi: 0.5401 (498 samples)
yi: 0.4294 (222 samples)
yo: 0.5116 (500 samples)
zh-CN: 0.2847 (488 samples)
zh-HK: 0.6063 (371 samples)
zh-TW: 0.3622 (466 samples)

Global average: 0.2362
```