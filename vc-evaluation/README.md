# VC Evaluation

## How to run generations

1. Download the audio,

```bash
wget https://huggingface.co/datasets/Scicom-intl/Evaluation-Multilingual-VC/resolve/main/vc_audio.zip
unzip vc_audio.zip -d common-voice
rm vc_audio.zip
```

### Dia TTS

```bash
python3 dia_tts.py --output 'dia-tts'
```

### Scicom Multilingual TTS

#### 0.6B

```bash
MODEL_NAME="Scicom-intl/Multilingual-TTS-0.6B-Base" python3 multilingual_tts.py --output 'multilingual-tts-0.6b'
```

#### 1.7B

```bash
MODEL_NAME="Scicom-intl/Multilingual-TTS-1.7B-Base" python3 multilingual_tts.py --output 'multilingual-tts-1.7b'
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

## How to calculate similarity

### Dia TTS

```bash
python3 calculate_similarity.py --output_folder "dia-tts" --output "dia-tts-similarity"
```

```
af: 0.3337 (130 samples)
am: 0.2528 (252 samples)
ar: 0.2600 (493 samples)
as: 0.3283 (379 samples)
az: 0.4802 (95 samples)
ba: 0.2114 (496 samples)
be: 0.3000 (500 samples)
bg: 0.1899 (500 samples)
bn: 0.3387 (500 samples)
br: 0.2780 (498 samples)
ca: 0.4604 (500 samples)
cs: 0.4725 (498 samples)
cy: 0.3492 (500 samples)
da: 0.4131 (498 samples)
de: 0.4162 (498 samples)
el: 0.1883 (498 samples)
en: 0.3945 (495 samples)
es: 0.5141 (495 samples)
et: 0.5140 (497 samples)
eu: 0.5320 (500 samples)
fa: 0.2358 (499 samples)
fi: 0.4246 (500 samples)
fr: 0.4552 (500 samples)
gl: 0.4735 (500 samples)
ha: 0.4259 (500 samples)
he: 0.2861 (392 samples)
hi: 0.2299 (500 samples)
ht: 0.3286 (5 samples)
hu: 0.4934 (500 samples)
hy-AM: 0.2462 (499 samples)
id: 0.4291 (500 samples)
is: 0.5780 (9 samples)
it: 0.4742 (500 samples)
ja: 0.2375 (414 samples)
ka: 0.3056 (500 samples)
kk: 0.2684 (500 samples)
ko: 0.2775 (458 samples)
lo: 0.4145 (26 samples)
lt: 0.5183 (500 samples)
lv: 0.4272 (500 samples)
mk: 0.1870 (499 samples)
ml: 0.2286 (490 samples)
mn: 0.2007 (500 samples)
mr: 0.3199 (500 samples)
mt: 0.4118 (500 samples)
ne-NP: 0.2240 (287 samples)
nl: 0.3822 (500 samples)
nn-NO: 0.3982 (423 samples)
oc: 0.3970 (274 samples)
pa-IN: 0.2699 (500 samples)
pl: 0.4170 (500 samples)
ps: 0.2219 (500 samples)
pt: 0.4123 (495 samples)
ro: 0.4659 (500 samples)
ru: 0.1558 (496 samples)
sd: 0.2208 (40 samples)
sk: 0.4039 (490 samples)
sl: 0.3981 (500 samples)
sq: 0.4697 (500 samples)
sr: 0.1547 (500 samples)
sv-SE: 0.3632 (500 samples)
sw: 0.5118 (500 samples)
ta: 0.3124 (500 samples)
te: 0.2680 (66 samples)
tg: 0.1748 (69 samples)
th: 0.2938 (496 samples)
tk: 0.4787 (497 samples)
tr: 0.3589 (500 samples)
tt: 0.2493 (498 samples)
uk: 0.1287 (496 samples)
ur: 0.2346 (500 samples)
uz: 0.4509 (500 samples)
vi: 0.3326 (496 samples)
yi: 0.2193 (222 samples)
yo: 0.4864 (500 samples)
zh-CN: 0.2815 (473 samples)
zh-HK: 0.2585 (279 samples)
zh-TW: 0.1461 (429 samples)

Global average: 0.3416
```

### Scicom Multilingual TTS

#### 0.6B

```bash
python3 calculate_similarity.py --output_folder "multilingual-tts-0.6b" --output "multilingual-tts-0.6b-similarity"
```

```
af: 0.5121 (131 samples)
am: 0.5132 (252 samples)
ar: 0.4300 (493 samples)
as: 0.5331 (379 samples)
az: 0.5316 (95 samples)
ba: 0.4551 (496 samples)
be: 0.5979 (500 samples)
bg: 0.6095 (500 samples)
bn: 0.5832 (500 samples)
br: 0.3314 (498 samples)
ca: 0.5070 (500 samples)
cs: 0.4890 (498 samples)
cy: 0.4149 (500 samples)
da: 0.4135 (498 samples)
de: 0.5412 (498 samples)
el: 0.4422 (500 samples)
en: 0.5087 (498 samples)
es: 0.5552 (495 samples)
et: 0.5698 (500 samples)
eu: 0.5635 (500 samples)
fa: 0.4468 (500 samples)
fi: 0.4590 (500 samples)
fr: 0.5369 (500 samples)
gl: 0.5093 (500 samples)
ha: 0.4805 (500 samples)
he: 0.4929 (392 samples)
hi: 0.4921 (500 samples)
ht: 0.3976 (5 samples)
hu: 0.5103 (500 samples)
hy-AM: 0.5831 (500 samples)
id: 0.4808 (500 samples)
is: 0.5248 (9 samples)
it: 0.5697 (500 samples)
ja: 0.4444 (416 samples)
ka: 0.5860 (500 samples)
kk: 0.4935 (500 samples)
ko: 0.5508 (458 samples)
lo: 0.4723 (26 samples)
lt: 0.5480 (500 samples)
lv: 0.4431 (500 samples)
mk: 0.5880 (500 samples)
ml: 0.4207 (490 samples)
mn: 0.5209 (500 samples)
mr: 0.5802 (500 samples)
mt: 0.4761 (500 samples)
ne-NP: 0.4764 (287 samples)
nl: 0.5041 (500 samples)
nn-NO: 0.4268 (423 samples)
oc: 0.4497 (274 samples)
pa-IN: 0.5042 (500 samples)
pl: 0.4855 (500 samples)
ps: 0.4846 (500 samples)
pt: 0.4039 (496 samples)
ro: 0.5348 (500 samples)
ru: 0.5169 (498 samples)
sd: 0.4272 (40 samples)
sk: 0.4075 (490 samples)
sl: 0.4358 (500 samples)
sq: 0.4804 (500 samples)
sr: 0.3240 (500 samples)
sv-SE: 0.3691 (500 samples)
sw: 0.5102 (500 samples)
ta: 0.5260 (500 samples)
te: 0.4663 (66 samples)
tg: 0.5160 (69 samples)
th: 0.4613 (496 samples)
tk: 0.4945 (498 samples)
tr: 0.3735 (500 samples)
tt: 0.4430 (500 samples)
uk: 0.4826 (500 samples)
ur: 0.5032 (500 samples)
uz: 0.5294 (500 samples)
vi: 0.3593 (496 samples)
yi: 0.4014 (222 samples)
yo: 0.5081 (500 samples)
zh-CN: 0.5086 (476 samples)
zh-HK: 0.5203 (282 samples)
zh-TW: 0.4256 (438 samples)

Global average: 0.4868
```

#### 1.7B

```bash
python3 calculate_similarity.py --output_folder "multilingual-tts-1.7b" --output "multilingual-tts-1.7b-similarity"
```

```
af: 0.5300 (131 samples)
am: 0.5480 (252 samples)
ar: 0.4484 (493 samples)
as: 0.5358 (379 samples)
az: 0.5675 (95 samples)
ba: 0.4666 (496 samples)
be: 0.6024 (500 samples)
bg: 0.6113 (500 samples)
bn: 0.5791 (500 samples)
br: 0.3648 (498 samples)
ca: 0.5173 (500 samples)
cs: 0.5014 (498 samples)
cy: 0.4528 (500 samples)
da: 0.4369 (498 samples)
de: 0.5347 (498 samples)
el: 0.4589 (500 samples)
en: 0.5124 (498 samples)
es: 0.5678 (495 samples)
et: 0.5877 (500 samples)
eu: 0.5648 (500 samples)
fa: 0.4871 (500 samples)
fi: 0.4907 (500 samples)
fr: 0.5380 (500 samples)
gl: 0.5273 (500 samples)
ha: 0.5105 (500 samples)
he: 0.4942 (392 samples)
hi: 0.4980 (500 samples)
ht: 0.4492 (5 samples)
hu: 0.5218 (500 samples)
hy-AM: 0.5819 (500 samples)
id: 0.4866 (500 samples)
is: 0.5673 (9 samples)
it: 0.5737 (500 samples)
ja: 0.4527 (416 samples)
ka: 0.5821 (500 samples)
kk: 0.5123 (500 samples)
ko: 0.5484 (458 samples)
lo: 0.5993 (26 samples)
lt: 0.5594 (500 samples)
lv: 0.4847 (500 samples)
mk: 0.5926 (500 samples)
ml: 0.4237 (490 samples)
mn: 0.5453 (500 samples)
mr: 0.5886 (500 samples)
mt: 0.4936 (500 samples)
ne-NP: 0.4876 (287 samples)
nl: 0.5031 (500 samples)
nn-NO: 0.4550 (423 samples)
oc: 0.4516 (274 samples)
pa-IN: 0.5041 (500 samples)
pl: 0.4995 (500 samples)
ps: 0.5196 (500 samples)
pt: 0.4211 (496 samples)
ro: 0.5545 (500 samples)
ru: 0.5239 (498 samples)
sd: 0.4301 (40 samples)
sk: 0.4262 (490 samples)
sl: 0.4579 (500 samples)
sq: 0.4986 (500 samples)
sr: 0.3531 (500 samples)
sv-SE: 0.3984 (500 samples)
sw: 0.5222 (500 samples)
ta: 0.5216 (500 samples)
te: 0.4552 (66 samples)
tg: 0.5320 (69 samples)
th: 0.4764 (496 samples)
tk: 0.5322 (498 samples)
tr: 0.3903 (500 samples)
tt: 0.4600 (500 samples)
uk: 0.4999 (500 samples)
ur: 0.5162 (500 samples)
uz: 0.5449 (500 samples)
vi: 0.3810 (496 samples)
yi: 0.4397 (222 samples)
yo: 0.5272 (500 samples)
zh-CN: 0.5115 (476 samples)
zh-HK: 0.5441 (282 samples)
zh-TW: 0.4428 (438 samples)

Global average: 0.5036
```

### Orpheus

```bash
python3 calculate_similarity.py --output_folder "orpheus" --output "orpheus-similarity"
```

```
af: 0.4950 (131 samples)
am: 0.3663 (252 samples)
ar: 0.2850 (493 samples)
as: 0.3605 (379 samples)
az: 0.4466 (95 samples)
ba: 0.3195 (343 samples)
be: 0.4125 (500 samples)
bg: 0.4543 (500 samples)
bn: 0.3445 (500 samples)
br: 0.3272 (498 samples)
ca: 0.4816 (500 samples)
cs: 0.4084 (498 samples)
cy: 0.3938 (500 samples)
da: 0.3953 (498 samples)
de: 0.4978 (498 samples)
el: 0.3316 (500 samples)
en: 0.4685 (498 samples)
es: 0.5430 (495 samples)
et: 0.5518 (500 samples)
eu: 0.5390 (500 samples)
fa: 0.3276 (500 samples)
fi: 0.4352 (500 samples)
fr: 0.4896 (500 samples)
gl: 0.4804 (500 samples)
ha: 0.4642 (500 samples)
he: 0.3426 (392 samples)
hi: 0.3370 (500 samples)
ht: 0.3526 (5 samples)
hu: 0.4881 (500 samples)
hy-AM: 0.3906 (500 samples)
id: 0.4497 (500 samples)
is: 0.5186 (9 samples)
it: 0.5479 (500 samples)
ja: 0.3311 (416 samples)
ka: 0.4027 (500 samples)
kk: 0.3570 (500 samples)
ko: 0.4240 (458 samples)
lo: 0.3593 (26 samples)
lt: 0.5255 (500 samples)
lv: 0.4343 (500 samples)
mk: 0.4379 (500 samples)
ml: 0.2859 (490 samples)
mn: 0.3823 (453 samples)
mr: 0.3434 (464 samples)
mt: 0.4367 (500 samples)
ne-NP: 0.3355 (287 samples)
nl: 0.4856 (500 samples)
nn-NO: 0.4093 (423 samples)
oc: 0.4280 (274 samples)
pa-IN: 0.3826 (500 samples)
pl: 0.4314 (500 samples)
ps: 0.3360 (500 samples)
pt: 0.3854 (496 samples)
ro: 0.5037 (500 samples)
ru: 0.3842 (498 samples)
sd: 0.2976 (40 samples)
sk: 0.3698 (490 samples)
sl: 0.4274 (500 samples)
sq: 0.4471 (500 samples)
sr: 0.2596 (500 samples)
sv-SE: 0.3503 (500 samples)
sw: 0.4772 (500 samples)
ta: 0.3173 (500 samples)
te: 0.3177 (66 samples)
tg: 0.3805 (69 samples)
th: 0.3184 (496 samples)
tk: 0.4201 (498 samples)
tr: 0.3165 (500 samples)
tt: 0.3313 (500 samples)
uk: 0.3516 (500 samples)
ur: 0.3167 (500 samples)
uz: 0.5045 (500 samples)
vi: 0.2812 (496 samples)
yi: 0.2832 (222 samples)
yo: 0.3770 (500 samples)
zh-CN: 0.4245 (476 samples)
zh-HK: 0.4177 (245 samples)
zh-TW: 0.3775 (438 samples)

Global average: 0.4002
```

### Chatterbox

```bash
python3 calculate_similarity.py --output_folder "chatterbox" --output "chatterbox-similarity"
```

```
ar: 0.6326 (493 samples)
da: 0.6584 (498 samples)
de: 0.7088 (498 samples)
el: 0.6365 (500 samples)
en: 0.6579 (498 samples)
es: 0.7312 (495 samples)
fi: 0.6916 (500 samples)
fr: 0.7068 (500 samples)
he: 0.6797 (392 samples)
hi: 0.6675 (500 samples)
it: 0.7334 (500 samples)
ja: 0.6203 (416 samples)
ko: 0.7314 (458 samples)
nl: 0.7002 (500 samples)
nn-NO: 0.6669 (423 samples)
pl: 0.6741 (500 samples)
pt: 0.6048 (496 samples)
ru: 0.6912 (498 samples)
sv-SE: 0.6264 (500 samples)
sw: 0.7075 (500 samples)
tr: 0.5785 (500 samples)
zh-CN: 0.6864 (476 samples)
zh-TW: 0.6282 (438 samples)

Global average: 0.6704
```

## How to calculate CER

### Dia TTS

```bash
python3 calculate_cer.py --output_folder "dia-tts" --output "dia-tts-cer"
```

```
af: 0.3029 (130 samples)
am: 0.9998 (252 samples)
ar: 0.9533 (493 samples)
as: 0.9901 (379 samples)
az: 0.5508 (95 samples)
ba: 0.9895 (496 samples)
be: 0.9419 (500 samples)
bg: 0.9449 (500 samples)
bn: 0.9789 (500 samples)
br: 0.6220 (498 samples)
ca: 0.2025 (500 samples)
cs: 0.4114 (498 samples)
cy: 0.4794 (500 samples)
da: 0.4549 (498 samples)
de: 0.2062 (498 samples)
el: 0.9690 (498 samples)
en: 0.1707 (495 samples)
es: 0.1019 (495 samples)
et: 0.3007 (497 samples)
eu: 0.1829 (500 samples)
fa: 0.9719 (499 samples)
fi: 0.3899 (500 samples)
fr: 0.2443 (500 samples)
gl: 0.2058 (500 samples)
ha: 0.3496 (500 samples)
he: 0.9576 (392 samples)
hi: 0.9757 (500 samples)
ht: 0.5741 (5 samples)
hu: 0.3764 (500 samples)
hy-AM: 0.9565 (499 samples)
id: 0.3050 (500 samples)
is: 0.3931 (9 samples)
it: 0.1502 (500 samples)
ja: 0.9863 (414 samples)
ka: 0.9712 (500 samples)
kk: 0.9796 (500 samples)
ko: 0.9720 (458 samples)
lo: 0.9972 (26 samples)
lt: 0.3816 (500 samples)
lv: 0.5298 (500 samples)
mk: 0.9514 (499 samples)
ml: 0.9919 (490 samples)
mn: 0.9845 (500 samples)
mr: 0.9645 (500 samples)
mt: 0.4463 (500 samples)
ne-NP: 0.9882 (287 samples)
nl: 0.1934 (500 samples)
nn-NO: 0.3921 (423 samples)
oc: 0.4540 (274 samples)
pa-IN: 0.9960 (500 samples)
pl: 0.5089 (500 samples)
ps: 0.9634 (500 samples)
pt: 0.4754 (495 samples)
ro: 0.3127 (500 samples)
ru: 0.9377 (496 samples)
sd: 0.9891 (40 samples)
sk: 0.7994 (490 samples)
sl: 0.4282 (500 samples)
sq: 0.4066 (500 samples)
sr: 0.9889 (500 samples)
sv-SE: 0.4409 (500 samples)
sw: 0.2750 (500 samples)
ta: 0.9731 (500 samples)
te: 0.9972 (66 samples)
tg: 0.9800 (69 samples)
th: 0.9910 (496 samples)
tk: 0.5880 (497 samples)
tr: 0.6333 (500 samples)
tt: 0.9784 (498 samples)
uk: 0.9571 (496 samples)
ur: 0.9531 (500 samples)
uz: 0.4078 (500 samples)
vi: 0.9794 (496 samples)
yi: 0.9791 (222 samples)
yo: 0.8336 (500 samples)
zh-CN: 0.9979 (473 samples)
zh-HK: 0.9997 (281 samples)
zh-TW: 0.9998 (429 samples)

Global average: 0.6867
```

### Scicom Multilingual TTS

#### 0.6B

```bash
python3 calculate_cer.py --output_folder "multilingual-tts-0.6b" --output "multilingual-tts-0.6b-cer"
```

```
af: 0.2587 (131 samples)
am: 1.0000 (252 samples)
ar: 0.2720 (493 samples)
as: 0.9216 (379 samples)
az: 0.2665 (95 samples)
ba: 0.8456 (496 samples)
be: 0.1466 (500 samples)
bg: 0.1593 (500 samples)
bn: 0.2635 (500 samples)
br: 0.5497 (498 samples)
ca: 0.1982 (500 samples)
cs: 0.2321 (498 samples)
cy: 0.5757 (500 samples)
da: 0.3154 (498 samples)
de: 0.0330 (498 samples)
el: 0.2938 (500 samples)
en: 0.0657 (498 samples)
es: 0.0966 (495 samples)
et: 0.2390 (500 samples)
eu: 0.1573 (500 samples)
fa: 0.3428 (500 samples)
fi: 0.2762 (500 samples)
fr: 0.0705 (500 samples)
gl: 0.1743 (500 samples)
ha: 0.3236 (500 samples)
he: 0.3846 (392 samples)
hi: 0.1595 (500 samples)
ht: 0.3943 (5 samples)
hu: 0.2915 (500 samples)
hy-AM: 0.1321 (500 samples)
id: 0.0632 (500 samples)
is: 0.4129 (9 samples)
it: 0.1390 (500 samples)
ja: 0.2748 (416 samples)
ka: 0.1377 (500 samples)
kk: 0.2805 (500 samples)
ko: 0.0816 (458 samples)
lo: 0.9996 (26 samples)
lt: 0.2221 (500 samples)
lv: 0.3340 (500 samples)
mk: 0.1070 (500 samples)
ml: 0.9664 (490 samples)
mn: 0.4968 (500 samples)
mr: 0.2113 (500 samples)
mt: 0.4565 (500 samples)
ne-NP: 0.3428 (287 samples)
nl: 0.1234 (500 samples)
nn-NO: 0.3561 (423 samples)
oc: 0.3979 (274 samples)
pa-IN: 0.4125 (500 samples)
pl: 0.2804 (500 samples)
ps: 0.4784 (500 samples)
pt: 0.3042 (496 samples)
ro: 0.1627 (500 samples)
ru: 0.1057 (498 samples)
sd: 0.9646 (40 samples)
sk: 0.4384 (490 samples)
sl: 0.2490 (500 samples)
sq: 0.3076 (500 samples)
sr: 0.7663 (500 samples)
sv-SE: 0.2865 (500 samples)
sw: 0.2568 (500 samples)
ta: 0.1771 (500 samples)
te: 0.6547 (66 samples)
tg: 0.3756 (69 samples)
th: 0.3245 (496 samples)
tk: 0.6428 (498 samples)
tr: 0.2749 (500 samples)
tt: 0.4033 (500 samples)
uk: 0.1861 (500 samples)
ur: 0.1311 (500 samples)
uz: 0.3643 (500 samples)
vi: 0.4255 (496 samples)
yi: 0.5936 (222 samples)
yo: 0.5849 (500 samples)
zh-CN: 0.2480 (476 samples)
zh-HK: 0.5884 (282 samples)
zh-TW: 0.4835 (438 samples)

Global average: 0.3502
```

#### 1.7B

```bash
python3 calculate_cer.py --output_folder "multilingual-tts-1.7b" --output "multilingual-tts-1.7b-cer"
```

```
af: 0.2057 (131 samples)
am: 1.0000 (252 samples)
ar: 0.2480 (493 samples)
as: 0.9227 (379 samples)
az: 0.1615 (95 samples)
ba: 0.8065 (496 samples)
be: 0.1207 (500 samples)
bg: 0.1197 (500 samples)
bn: 0.2563 (500 samples)
br: 0.4922 (498 samples)
ca: 0.1458 (500 samples)
cs: 0.1546 (498 samples)
cy: 0.4886 (500 samples)
da: 0.2551 (498 samples)
de: 0.0260 (498 samples)
el: 0.2173 (500 samples)
en: 0.0457 (498 samples)
es: 0.0571 (495 samples)
et: 0.1750 (500 samples)
eu: 0.1325 (500 samples)
fa: 0.2308 (500 samples)
fi: 0.2100 (500 samples)
fr: 0.0703 (500 samples)
gl: 0.1304 (500 samples)
ha: 0.2598 (500 samples)
he: 0.3292 (392 samples)
hi: 0.1625 (500 samples)
ht: 0.4319 (5 samples)
hu: 0.2238 (500 samples)
hy-AM: 0.1221 (500 samples)
id: 0.0501 (500 samples)
is: 0.2425 (9 samples)
it: 0.0704 (500 samples)
ja: 0.2070 (416 samples)
ka: 0.1342 (500 samples)
kk: 0.2141 (500 samples)
ko: 0.0828 (458 samples)
lo: 0.9982 (26 samples)
lt: 0.1801 (500 samples)
lv: 0.2474 (500 samples)
mk: 0.0872 (500 samples)
ml: 0.9626 (490 samples)
mn: 0.3269 (500 samples)
mr: 0.2093 (500 samples)
mt: 0.3445 (500 samples)
ne-NP: 0.3208 (287 samples)
nl: 0.0846 (500 samples)
nn-NO: 0.2536 (423 samples)
oc: 0.3718 (274 samples)
pa-IN: 0.3924 (500 samples)
pl: 0.1746 (500 samples)
ps: 0.4081 (500 samples)
pt: 0.2327 (496 samples)
ro: 0.0905 (500 samples)
ru: 0.0925 (498 samples)
sd: 0.9383 (40 samples)
sk: 0.3890 (490 samples)
sl: 0.1952 (500 samples)
sq: 0.2239 (500 samples)
sr: 0.7674 (500 samples)
sv-SE: 0.2310 (500 samples)
sw: 0.1870 (500 samples)
ta: 0.1728 (500 samples)
te: 0.5880 (66 samples)
tg: 0.3008 (69 samples)
th: 0.2578 (496 samples)
tk: 0.5732 (498 samples)
tr: 0.2639 (500 samples)
tt: 0.3657 (500 samples)
uk: 0.1182 (500 samples)
ur: 0.1211 (500 samples)
uz: 0.3096 (500 samples)
vi: 0.3570 (496 samples)
yi: 0.4780 (222 samples)
yo: 0.5375 (500 samples)
zh-CN: 0.2165 (476 samples)
zh-HK: 0.4680 (282 samples)
zh-TW: 0.4133 (438 samples)

Global average: 0.3007
```

### Orpheus

```bash
python3 calculate_cer.py --output_folder "orpheus" --output "orpheus-cer"
```

```
af: 0.3086 (131 samples)
am: 0.9982 (252 samples)
ar: 0.9274 (493 samples)
as: 0.9657 (379 samples)
az: 0.7177 (95 samples)
ba: 0.9655 (343 samples)
be: 0.8430 (500 samples)
bg: 0.7824 (500 samples)
bn: 0.9459 (500 samples)
br: 0.5855 (498 samples)
ca: 0.2765 (500 samples)
cs: 0.7404 (498 samples)
cy: 0.6102 (500 samples)
da: 0.4519 (498 samples)
de: 0.1483 (498 samples)
el: 0.9472 (500 samples)
en: 0.1021 (498 samples)
es: 0.1900 (495 samples)
et: 0.3801 (500 samples)
eu: 0.1949 (500 samples)
fa: 0.8795 (500 samples)
fi: 0.3708 (500 samples)
fr: 0.3298 (500 samples)
gl: 0.2465 (500 samples)
ha: 0.3852 (500 samples)
he: 0.8824 (392 samples)
hi: 0.8939 (500 samples)
ht: 0.4196 (5 samples)
hu: 0.4544 (500 samples)
hy-AM: 0.9008 (500 samples)
id: 0.3449 (500 samples)
is: 0.4379 (9 samples)
it: 0.2140 (500 samples)
ja: 0.9673 (416 samples)
ka: 0.9234 (500 samples)
kk: 0.8672 (500 samples)
ko: 0.9239 (458 samples)
lo: 0.9994 (26 samples)
lt: 0.3732 (500 samples)
lv: 0.4114 (500 samples)
mk: 0.8200 (500 samples)
ml: 0.9826 (490 samples)
mn: 0.9270 (453 samples)
mr: 0.9003 (464 samples)
mt: 0.6079 (500 samples)
ne-NP: 0.9144 (287 samples)
nl: 0.2345 (500 samples)
nn-NO: 0.4144 (423 samples)
oc: 0.4571 (274 samples)
pa-IN: 0.9653 (500 samples)
pl: 0.6321 (500 samples)
ps: 0.8765 (500 samples)
pt: 0.4555 (496 samples)
ro: 0.3581 (500 samples)
ru: 0.8895 (498 samples)
sd: 0.9601 (40 samples)
sk: 0.7087 (490 samples)
sl: 0.3687 (500 samples)
sq: 0.5085 (500 samples)
sr: 0.9196 (500 samples)
sv-SE: 0.4179 (500 samples)
sw: 0.3484 (500 samples)
ta: 0.8962 (500 samples)
te: 0.9858 (66 samples)
tg: 0.8695 (69 samples)
th: 0.9619 (496 samples)
tk: 0.7387 (498 samples)
tr: 0.7374 (500 samples)
tt: 0.8962 (500 samples)
uk: 0.9032 (500 samples)
ur: 0.8716 (500 samples)
uz: 0.4403 (500 samples)
vi: 0.9291 (496 samples)
yi: 0.8958 (222 samples)
yo: 0.8706 (500 samples)
zh-CN: 0.9475 (476 samples)
zh-HK: 0.9697 (245 samples)
zh-TW: 0.9227 (438 samples)

Global average: 0.6771
```

### Chatterbox

```bash
python3 calculate_cer.py --output_folder "chatterbox" --output "chatterbox-cer"
```

```
ar: 0.1577 (493 samples)
da: 0.0956 (498 samples)
de: 0.0434 (498 samples)
el: 0.1077 (500 samples)
en: 0.0554 (498 samples)
es: 0.0318 (495 samples)
fi: 0.0491 (500 samples)
fr: 0.0814 (500 samples)
he: 0.3864 (392 samples)
hi: 0.1644 (500 samples)
it: 0.0313 (500 samples)
ja: 0.1669 (416 samples)
ko: 0.0651 (458 samples)
nl: 0.0153 (500 samples)
nn-NO: 0.0943 (423 samples)
pl: 0.0465 (500 samples)
pt: 0.0933 (496 samples)
ru: 0.0447 (498 samples)
sv-SE: 0.0415 (500 samples)
sw: 0.1117 (500 samples)
tr: 0.0692 (500 samples)
zh-CN: 0.2252 (476 samples)
zh-TW: 0.3508 (438 samples)

Global average: 0.1099
```