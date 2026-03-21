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

```bash
python3 chatterbox.py --output 'chatterbox'
```

### FishSpeech2

```bash
python3 fishspeech2.py --output 'fishspeech2'
```

### Qwen3 TTS

```bash
python3 qwen3_tts.py --output 'qwen3_tts'
```

## How to calculate CER

### Dia TTS

```bash
python3 calculate_cer.py --output_folder "dia-tts" --output "dia-tts-cer"
```

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

### Orpheus

```bash
python3 calculate_cer.py --output_folder "orpheus" --output "orpheus-cer"
```

```
af: 0.2249 (131 samples)
am: 0.9887 (252 samples)
ar: 0.8651 (496 samples)
as: 0.9586 (379 samples)
az: 0.6008 (95 samples)
ba: 0.9341 (498 samples)
be: 0.7796 (500 samples)
bg: 0.7500 (500 samples)
bn: 0.9211 (500 samples)
br: 0.3763 (499 samples)
ca: 0.1740 (500 samples)
cs: 0.5814 (499 samples)
cy: 0.4361 (500 samples)
da: 0.3193 (499 samples)
de: 0.1540 (499 samples)
el: 0.9048 (500 samples)
en: 0.0252 (499 samples)
es: 0.1179 (497 samples)
et: 0.3336 (500 samples)
eu: 0.1610 (500 samples)
fa: 0.8327 (500 samples)
fi: 0.2934 (500 samples)
fr: 0.2076 (500 samples)
gl: 0.1789 (500 samples)
ha: 0.2479 (500 samples)
he: 0.8784 (392 samples)
hi: 0.8604 (500 samples)
ht: 0.2889 (5 samples)
hu: 0.3713 (500 samples)
hy-AM: 0.8983 (500 samples)
id: 0.2553 (500 samples)
is: 0.3457 (9 samples)
it: 0.1342 (500 samples)
ja: 0.9551 (456 samples)
ka: 0.8865 (500 samples)
kk: 0.8094 (500 samples)
ko: 0.8987 (465 samples)
lo: 0.9967 (26 samples)
lt: 0.3215 (500 samples)
lv: 0.2943 (500 samples)
mk: 0.7425 (500 samples)
ml: 0.9779 (495 samples)
mn: 0.8454 (500 samples)
mr: 0.8642 (500 samples)
mt: 0.4974 (500 samples)
ne-NP: 0.8778 (287 samples)
nl: 0.1837 (500 samples)
nn-NO: 0.3122 (423 samples)
oc: 0.3432 (274 samples)
pa-IN: 0.9704 (500 samples)
pl: 0.5378 (500 samples)
ps: 0.8402 (500 samples)
pt: 0.2542 (498 samples)
ro: 0.2731 (500 samples)
ru: 0.8244 (499 samples)
sd: 0.9393 (40 samples)
sk: 0.5690 (495 samples)
sl: 0.2637 (500 samples)
sq: 0.4161 (500 samples)
sr: 0.8659 (500 samples)
sv-SE: 0.2938 (500 samples)
sw: 0.1940 (500 samples)
ta: 0.8891 (500 samples)
te: 0.9677 (66 samples)
tg: 0.7831 (69 samples)
th: 0.9585 (498 samples)
tk: 0.6230 (499 samples)
tr: 0.5467 (500 samples)
tt: 0.8062 (500 samples)
uk: 0.8386 (500 samples)
ur: 0.8454 (500 samples)
uz: 0.3247 (500 samples)
vi: 0.8883 (498 samples)
yi: 0.8660 (222 samples)
yo: 0.7384 (500 samples)
zh-CN: 0.9462 (488 samples)
zh-HK: 0.9848 (371 samples)
zh-TW: 0.9294 (466 samples)

Global average: 0.6075
```

### Chatterbox

```bash
python3 calculate_cer.py --output_folder "chatterbox" --output "chatterbox-cer"
```

### FishSpeech2

```bash
python3 calculate_cer.py --output_folder "fishspeech2" --output "fishspeech2-cer"
```

```
af: 0.1038 (131 samples)
am: 1.0000 (252 samples)
ar: 0.1338 (496 samples)
as: 0.9265 (379 samples)
az: 0.1028 (95 samples)
ba: 0.5808 (498 samples)
be: 0.0997 (500 samples)
bg: 0.0926 (500 samples)
bn: 0.2325 (500 samples)
br: 0.3175 (499 samples)
ca: 0.0769 (500 samples)
cs: 0.0757 (499 samples)
cy: 0.3031 (500 samples)
da: 0.1238 (499 samples)
de: 0.0178 (499 samples)
el: 0.2354 (500 samples)
en: 0.0197 (499 samples)
es: 0.0225 (497 samples)
et: 0.0637 (500 samples)
eu: 0.0749 (500 samples)
fa: 0.1236 (500 samples)
fi: 0.0636 (500 samples)
fr: 0.0497 (500 samples)
gl: 0.0683 (500 samples)
ha: 0.2102 (500 samples)
he: 0.2891 (392 samples)
hi: 0.0981 (500 samples)
ht: 0.2029 (5 samples)
hu: 0.0680 (500 samples)
hy-AM: 0.2037 (500 samples)
id: 0.0372 (500 samples)
is: 0.1896 (9 samples)
it: 0.0333 (500 samples)
ja: 0.1492 (456 samples)
ka: 0.1462 (500 samples)
kk: 0.1577 (500 samples)
ko: 0.0456 (465 samples)
lo: 0.9989 (26 samples)
lt: 0.1265 (500 samples)
lv: 0.1283 (500 samples)
mk: 0.0958 (500 samples)
ml: 0.9635 (495 samples)
mn: 0.2140 (500 samples)
mr: 0.2013 (500 samples)
mt: 0.2707 (500 samples)
ne-NP: 0.2511 (287 samples)
nl: 0.0228 (500 samples)
nn-NO: 0.1290 (423 samples)
oc: 0.2499 (274 samples)
pa-IN: 0.4403 (500 samples)
pl: 0.0801 (500 samples)
ps: 0.3546 (500 samples)
pt: 0.0813 (498 samples)
ro: 0.0773 (500 samples)
ru: 0.0302 (499 samples)
sd: 0.9481 (40 samples)
sk: 0.2258 (495 samples)
sl: 0.0747 (500 samples)
sq: 0.2010 (500 samples)
sr: 0.6282 (500 samples)
sv-SE: 0.0466 (500 samples)
sw: 0.1489 (500 samples)
ta: 0.1118 (500 samples)
te: 0.7638 (66 samples)
tg: 0.2518 (69 samples)
th: 0.1427 (498 samples)
tk: 0.4027 (499 samples)
tr: 0.0614 (500 samples)
tt: 0.2481 (500 samples)
uk: 0.0765 (500 samples)
ur: 0.1004 (500 samples)
uz: 0.2351 (500 samples)
vi: 0.3362 (498 samples)
yi: 0.3707 (222 samples)
yo: 0.5057 (500 samples)
zh-CN: 0.1806 (488 samples)
zh-HK: 0.5295 (371 samples)
zh-TW: 0.4368 (466 samples)

Global average: 0.2370
```