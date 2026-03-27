"""
plot_results.py
───────────────
Generates a PNG heatmap and scatter plot comparing TTS models across languages
using Character Error Rate (CER).

To add a new model:   add an entry to MODELS and a row to CER_DATA.
To change style:      edit the STYLE dict.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# ══════════════════════════════════════════════════════════════════════════════
#  LANGUAGES  (column order — edit to reorder or add new ones)
# ══════════════════════════════════════════════════════════════════════════════
LANGS = [
    'af',    'am',    'ar',    'as',    'az',    'ba',    'be',    'bg',
    'bn',    'br',    'ca',    'cs',    'cy',    'da',    'de',    'el',
    'en',    'es',    'et',    'eu',    'fa',    'fi',    'fr',    'gl',
    'ha',    'he',    'hi',    'ht',    'hu',    'hy-AM', 'id',    'is',
    'it',    'ja',    'ka',    'kk',    'ko',    'lo',    'lt',    'lv',
    'mk',    'ml',    'mn',    'mr',    'mt',    'ne-NP', 'nl',    'nn-NO',
    'oc',    'pa-IN', 'pl',    'ps',    'pt',    'ro',    'ru',    'sd',
    'sk',    'sl',    'sq',    'sr',    'sv-SE', 'sw',    'ta',    'te',
    'tg',    'th',    'tk',    'tr',    'tt',    'uk',    'ur',    'uz',
    'vi',    'yi',    'yo',    'zh-CN', 'zh-HK', 'zh-TW',
]

# ══════════════════════════════════════════════════════════════════════════════
#  MODELS  (row order — edit display name)
# ══════════════════════════════════════════════════════════════════════════════
MODELS = [
    'Dia TTS',
    'Multilingual-Expressive-TTS-0.6B',
    'Multilingual-Expressive-TTS-1.7B',
    'Orpheus',
    'Chatterbox',
    'Fish Audio S2 Pro',
    'Qwen3 TTS',
]

# ══════════════════════════════════════════════════════════════════════════════
#  DATA  ── each row is a dict {lang_code: score}
#          missing keys → shown as "–" in the plot
#          language codes must match entries in LANGS exactly
# ══════════════════════════════════════════════════════════════════════════════

CER_DATA = [
    # ── Dia TTS ──────────────────────────────────────────────────────────────
    {
        'af':0.4801,'am':0.9999,'ar':0.9409,'as':0.9938,'az':0.7849,
        'ba':0.9957,'be':0.9744,'bg':0.9671,'bn':0.9948,'br':0.8235,
        'ca':0.4958,'cs':0.6881,'cy':0.6765,'da':0.6796,'de':0.4414,
        'el':0.9705,'en':0.4348,'es':0.3549,'et':0.4166,'eu':0.4615,
        'fa':0.9739,'fi':0.6843,'fr':0.4690,'gl':0.5142,'ha':0.6598,
        'he':0.9674,'hi':0.9854,'ht':0.8986,'hu':0.6499,'hy-AM':0.9898,
        'id':0.5699,'is':0.6973,'it':0.3556,'ja':0.9887,'ka':0.9844,
        'kk':0.9867,'ko':0.9831,'lo':1.0000,'lt':0.6421,'lv':0.7637,
        'mk':0.9786,'ml':0.9949,'mn':0.9965,'mr':0.9930,'mt':0.6834,
        'ne-NP':0.9944,'nl':0.4600,'nn-NO':0.6656,'oc':0.7252,'pa-IN':0.9978,
        'pl':0.7334,'ps':0.9714,'pt':0.6718,'ro':0.6455,'ru':0.9420,
        'sd':0.9956,'sk':0.8881,'sl':0.7402,'sq':0.7036,'sr':0.9927,
        'sv-SE':0.7190,'sw':0.5720,'ta':0.9926,'te':1.0000,'tg':0.9797,
        'th':0.9966,'tk':0.8167,'tr':0.7911,'tt':0.9840,'uk':0.9426,
        'ur':0.9617,'uz':0.6420,'vi':0.9840,'yi':0.9894,'yo':0.9396,
        'zh-CN':0.9994,'zh-HK':0.9997,'zh-TW':0.9995,
        '__avg__': 0.8131,
    },
    # ── Multilingual-Expressive-TTS-0.6B ────────────────────────────────────────────────
    {
        'af':0.1346,'am':1.0000,'ar':0.1417,'as':0.9192,'az':0.0416,
        'ba':0.6995,'be':0.0831,'bg':0.1094,'bn':0.3311,'br':0.3311,
        'ca':0.0740,'cs':0.0651,'cy':0.2840,'da':0.1472,'de':0.0197,
        'el':0.1017,'en':0.0200,'es':0.0265,'et':0.0972,'eu':0.0817,
        'fa':0.1138,'fi':0.1024,'fr':0.0511,'gl':0.0698,'ha':0.2233,
        'he':0.2154,'hi':0.0727,'ht':0.1825,'hu':0.0864,'hy-AM':0.0886,
        'id':0.0228,'is':0.2057,'it':0.0418,'ja':0.3158,'ka':0.0962,
        'kk':0.1208,'ko':0.0486,'lo':0.9977,'lt':0.0929,'lv':0.1410,
        'mk':0.0960,'ml':0.9627,'mn':0.1959,'mr':0.1605,'mt':0.2782,
        'ne-NP':0.2093,'nl':0.0338,'nn-NO':0.1870,'oc':0.2786,'pa-IN':0.3422,
        'pl':0.0906,'ps':0.3355,'pt':0.0806,'ro':0.0449,'ru':0.0388,
        'sd':0.9323,'sk':0.2118,'sl':0.0951,'sq':0.1794,'sr':0.6302,
        'sv-SE':0.0936,'sw':0.1100,'ta':0.0924,'te':0.5294,'tg':0.2685,
        'th':0.1394,'tk':0.4219,'tr':0.0754,'tt':0.2748,'uk':0.0732,
        'ur':0.0628,'uz':0.2579,'vi':0.5255,'yi':0.4113,'yo':0.5187,
        'zh-CN':0.2820,'zh-HK':0.6496,'zh-TW':0.4257,
        '__avg__': 0.2384,
    },
    # ── Multilingual-Expressive-TTS-1.7B ────────────────────────────────────────────────
    {
        'af':0.1158,'am':1.0000,'ar':0.1384,'as':0.9360,'az':0.0342,
        'ba':0.6757,'be':0.0812,'bg':0.1035,'bn':0.3762,'br':0.3344,
        'ca':0.0751,'cs':0.0728,'cy':0.2776,'da':0.1586,'de':0.0186,
        'el':0.1052,'en':0.0230,'es':0.0244,'et':0.0892,'eu':0.0797,
        'fa':0.1260,'fi':0.0989,'fr':0.0517,'gl':0.0688,'ha':0.2299,
        'he':0.2303,'hi':0.0707,'ht':0.1874,'hu':0.0842,'hy-AM':0.0948,
        'id':0.0215,'is':0.1868,'it':0.0369,'ja':0.3074,'ka':0.1074,
        'kk':0.1019,'ko':0.0483,'lo':0.9958,'lt':0.0934,'lv':0.1239,
        'mk':0.0906,'ml':0.9625,'mn':0.1934,'mr':0.1586,'mt':0.2769,
        'ne-NP':0.2154,'nl':0.0216,'nn-NO':0.1951,'oc':0.2707,'pa-IN':0.3409,
        'pl':0.0744,'ps':0.3385,'pt':0.0798,'ro':0.0423,'ru':0.0381,
        'sd':0.9308,'sk':0.2215,'sl':0.1029,'sq':0.1697,'sr':0.6588,
        'sv-SE':0.1044,'sw':0.1055,'ta':0.0887,'te':0.5025,'tg':0.2542,
        'th':0.1681,'tk':0.3701,'tr':0.0619,'tt':0.2657,'uk':0.0722,
        'ur':0.0654,'uz':0.2629,'vi':0.5401,'yi':0.4294,'yo':0.5116,
        'zh-CN':0.2847,'zh-HK':0.6063,'zh-TW':0.3622,
        '__avg__': 0.2362,
    },
    # ── Orpheus ──────────────────────────────────────────────────────────────
    {
        'af':0.2249,'am':0.9887,'ar':0.8651,'as':0.9586,'az':0.6008,
        'ba':0.9341,'be':0.7796,'bg':0.7500,'bn':0.9211,'br':0.3763,
        'ca':0.1740,'cs':0.5814,'cy':0.4361,'da':0.3193,'de':0.1540,
        'el':0.9048,'en':0.0252,'es':0.1179,'et':0.3336,'eu':0.1610,
        'fa':0.8327,'fi':0.2934,'fr':0.2076,'gl':0.1789,'ha':0.2479,
        'he':0.8784,'hi':0.8604,'ht':0.2889,'hu':0.3713,'hy-AM':0.8983,
        'id':0.2553,'is':0.3457,'it':0.1342,'ja':0.9551,'ka':0.8865,
        'kk':0.8094,'ko':0.8987,'lo':0.9967,'lt':0.3215,'lv':0.2943,
        'mk':0.7425,'ml':0.9779,'mn':0.8454,'mr':0.8642,'mt':0.4974,
        'ne-NP':0.8778,'nl':0.1837,'nn-NO':0.3122,'oc':0.3432,'pa-IN':0.9704,
        'pl':0.5378,'ps':0.8402,'pt':0.2542,'ro':0.2731,'ru':0.8244,
        'sd':0.9393,'sk':0.5690,'sl':0.2637,'sq':0.4161,'sr':0.8659,
        'sv-SE':0.2938,'sw':0.1940,'ta':0.8891,'te':0.9677,'tg':0.7831,
        'th':0.9585,'tk':0.6230,'tr':0.5467,'tt':0.8062,'uk':0.8386,
        'ur':0.8454,'uz':0.3247,'vi':0.8883,'yi':0.8660,'yo':0.7384,
        'zh-CN':0.9462,'zh-HK':0.9848,'zh-TW':0.9294,
        '__avg__': 0.6075,
    },
    # ── Chatterbox  (only 23 languages) ──────────────────────────────────────
    {
        'ar':0.2572,'da':0.1269,'de':0.2234,'el':0.1328,'en':0.1215,
        'es':0.1015,'fi':0.0854,'fr':0.1931,'he':0.5100,'hi':0.2845,
        'it':0.0620,'ja':0.2244,'ko':0.0826,'nl':0.0975,'nn-NO':0.1204,
        'pl':0.0780,'pt':0.1573,'ru':0.0647,'sv-SE':0.0811,'sw':0.1531,
        'tr':0.1195,'zh-CN':0.2912,'zh-TW':0.3377,
        '__avg__': 0.1698,
    },
    # ── Fish Audio S2 Pro ──────────────────────────────────────────────────────────
    {
        'af':0.1038,'am':1.0000,'ar':0.1338,'as':0.9265,'az':0.1028,
        'ba':0.5808,'be':0.0997,'bg':0.0926,'bn':0.2325,'br':0.3175,
        'ca':0.0769,'cs':0.0757,'cy':0.3031,'da':0.1238,'de':0.0178,
        'el':0.2354,'en':0.0197,'es':0.0225,'et':0.0637,'eu':0.0749,
        'fa':0.1236,'fi':0.0636,'fr':0.0497,'gl':0.0683,'ha':0.2102,
        'he':0.2891,'hi':0.0981,'ht':0.2029,'hu':0.0680,'hy-AM':0.2037,
        'id':0.0372,'is':0.1896,'it':0.0333,'ja':0.1492,'ka':0.1462,
        'kk':0.1577,'ko':0.0456,'lo':0.9989,'lt':0.1265,'lv':0.1283,
        'mk':0.0958,'ml':0.9635,'mn':0.2140,'mr':0.2013,'mt':0.2707,
        'ne-NP':0.2511,'nl':0.0228,'nn-NO':0.1290,'oc':0.2499,'pa-IN':0.4403,
        'pl':0.0801,'ps':0.3546,'pt':0.0813,'ro':0.0773,'ru':0.0302,
        'sd':0.9481,'sk':0.2258,'sl':0.0747,'sq':0.2010,'sr':0.6282,
        'sv-SE':0.0466,'sw':0.1489,'ta':0.1118,'te':0.7638,'tg':0.2518,
        'th':0.1427,'tk':0.4027,'tr':0.0614,'tt':0.2481,'uk':0.0765,
        'ur':0.1004,'uz':0.2351,'vi':0.3362,'yi':0.3707,'yo':0.5057,
        'zh-CN':0.1806,'zh-HK':0.5295,'zh-TW':0.4368,
        '__avg__': 0.2370,
    },
    # ── Qwen3 TTS  (only 11 languages) ───────────────────────────────────────
    {
        'de':0.0218,'en':0.0292,'es':0.0326,'fr':0.0583,'it':0.0435,
        'ja':0.2031,'ko':0.0532,'pt':0.1362,'ru':0.0502,
        'zh-CN':0.1624,'zh-TW':0.3800,
        '__avg__': 0.1064,
    },
]

# ══════════════════════════════════════════════════════════════════════════════
#  MOS DATA  ── Mean Opinion Score (UTMOSv2, 1–5 scale)
#              missing models → all NaN cells + NaN avg
# ══════════════════════════════════════════════════════════════════════════════
_nan = float('nan')

MOS_DATA = [
    # ── Dia TTS ───────────────────────────────────────────────────────────────
    {
        'af':2.2471,'am':1.7337,'ar':1.7033,'as':1.7492,'az':1.9577,
        'ba':1.7044,'be':1.7727,'bg':1.5621,'bn':1.8448,'br':1.6231,
        'ca':2.0167,'cs':1.9180,'cy':1.9149,'da':1.9296,'de':2.1887,
        'el':1.7758,'en':1.9760,'es':2.1636,'et':2.4030,'eu':2.3006,
        'fa':1.6383,'fi':1.8968,'fr':2.0537,'gl':2.0420,'ha':1.8402,
        'he':1.7136,'hi':1.6255,'ht':1.9812,'hu':2.0455,'hy-AM':1.7129,
        'id':2.0662,'is':2.0062,'it':2.2322,'ja':1.8723,'ka':1.8551,
        'kk':1.6974,'ko':1.5789,'lo':1.8653,'lt':2.1388,'lv':1.8705,
        'mk':1.5074,'ml':1.8631,'mn':1.5779,'mr':1.7815,'mt':2.0890,
        'ne-NP':1.7211,'nl':2.1065,'nn-NO':1.8905,'oc':1.8265,'pa-IN':1.7590,
        'pl':1.8804,'ps':1.5740,'pt':1.9530,'ro':1.9428,'ru':1.6516,
        'sd':1.6618,'sk':1.8710,'sl':1.7868,'sq':1.9454,'sr':1.5622,
        'sv-SE':1.8089,'sw':2.2197,'ta':1.8343,'te':1.8003,'tg':1.4999,
        'th':1.8742,'tk':1.9589,'tr':1.8553,'tt':1.7529,'uk':1.6260,
        'ur':1.5768,'uz':2.1818,'vi':1.5317,'yi':1.7104,'yo':1.7179,
        'zh-CN':1.8971,'zh-HK':1.8981,'zh-TW':1.9719,
        '__avg__': 1.8575,
    },
    # ── Multilingual-Expressive-TTS-0.6B ──────────────────────────────────────
    {
        'af':3.3333,'am':3.2761,'ar':3.1850,'as':3.2322,'az':3.1916,
        'ba':3.0963,'be':3.2335,'bg':3.2341,'bn':3.2044,'br':3.1423,
        'ca':3.2201,'cs':3.2258,'cy':3.2941,'da':3.3018,'de':3.2963,
        'el':3.2046,'en':3.5712,'es':3.2050,'et':3.2800,'eu':3.2379,
        'fa':3.2019,'fi':3.2996,'fr':3.1638,'gl':3.2531,'ha':3.3168,
        'he':3.2590,'hi':3.1946,'ht':3.2869,'hu':3.2493,'hy-AM':3.1522,
        'id':3.2205,'is':3.2593,'it':3.2260,'ja':3.1508,'ka':3.1751,
        'kk':3.1333,'ko':3.0851,'lo':3.2109,'lt':3.2714,'lv':3.2486,
        'mk':3.2366,'ml':3.1769,'mn':3.1648,'mr':3.1820,'mt':3.2862,
        'ne-NP':3.2393,'nl':3.3584,'nn-NO':3.3027,'oc':3.2293,'pa-IN':3.2285,
        'pl':3.2618,'ps':3.2188,'pt':3.2072,'ro':3.2615,'ru':3.1882,
        'sd':3.2792,'sk':3.1526,'sl':3.2194,'sq':3.2027,'sr':3.0163,
        'sv-SE':3.2189,'sw':3.3415,'ta':3.1910,'te':3.2538,'tg':3.1549,
        'th':3.2146,'tk':3.2020,'tr':3.1076,'tt':3.1428,'uk':3.2035,
        'ur':3.2268,'uz':3.2229,'vi':3.2520,'yi':3.2762,'yo':3.2922,
        'zh-CN':3.2257,'zh-HK':3.2032,'zh-TW':3.2620,
        '__avg__': 3.2273,
    },
    # ── Multilingual-Expressive-TTS-1.7B ──────────────────────────────────────
    {
        'af':3.3340,'am':3.2847,'ar':3.1970,'as':3.2721,'az':3.2355,
        'ba':3.0950,'be':3.2159,'bg':3.2276,'bn':3.2457,'br':3.1386,
        'ca':3.2539,'cs':3.2480,'cy':3.3059,'da':3.3080,'de':3.2864,
        'el':3.2021,'en':3.5550,'es':3.1922,'et':3.3047,'eu':3.2556,
        'fa':3.2031,'fi':3.3064,'fr':3.1835,'gl':3.2519,'ha':3.3397,
        'he':3.2859,'hi':3.1871,'ht':3.1869,'hu':3.2711,'hy-AM':3.1496,
        'id':3.2195,'is':3.3052,'it':3.2313,'ja':3.1533,'ka':3.2063,
        'kk':3.1151,'ko':3.1004,'lo':3.2337,'lt':3.2864,'lv':3.2288,
        'mk':3.2285,'ml':3.1492,'mn':3.1630,'mr':3.1915,'mt':3.3114,
        'ne-NP':3.2047,'nl':3.3422,'nn-NO':3.3147,'oc':3.2672,'pa-IN':3.2304,
        'pl':3.2262,'ps':3.2384,'pt':3.1941,'ro':3.2850,'ru':3.1888,
        'sd':3.2637,'sk':3.1466,'sl':3.2096,'sq':3.2484,'sr':2.9958,
        'sv-SE':3.2233,'sw':3.3559,'ta':3.2191,'te':3.2042,'tg':3.1391,
        'th':3.2309,'tk':3.2263,'tr':3.1340,'tt':3.1505,'uk':3.1967,
        'ur':3.2193,'uz':3.2426,'vi':3.2658,'yi':3.3105,'yo':3.3649,
        'zh-CN':3.2459,'zh-HK':3.2052,'zh-TW':3.2347,
        '__avg__': 3.2330,
    },
    # ── Orpheus ───────────────────────────────────────────────────────────────
    {
        'af':3.1350,'am':2.1176,'ar':2.4434,'as':2.4000,'az':2.8158,
        'ba':2.4047,'be':2.4632,'bg':2.5660,'bn':2.3831,'br':2.9996,
        'ca':3.0616,'cs':2.8615,'cy':3.0697,'da':3.0477,'de':3.0973,
        'el':2.5823,'en':3.3825,'es':3.1008,'et':3.0651,'eu':3.1237,
        'fa':2.4646,'fi':3.0501,'fr':2.9805,'gl':3.0643,'ha':3.0007,
        'he':2.2192,'hi':2.4366,'ht':2.8506,'hu':3.0643,'hy-AM':2.3231,
        'is':2.8223,'it':3.1072,'ja':2.6027,'ka':2.4252,'kk':2.4622,
        'ko':2.4408,'lo':2.3549,'lt':3.0585,'lv':2.9755,'mk':2.5562,
        'ml':2.3149,'mn':2.4466,'mr':2.4417,'mt':2.8775,'ne-NP':2.4219,
        'nl':3.1561,'nn-NO':3.0390,'oc':2.9822,'pa-IN':2.7234,'pl':2.9834,
        'ps':2.3386,'pt':3.0628,'ro':3.0140,'ru':2.6044,'sd':2.2748,
        'sk':2.9829,'sl':2.9918,'sq':2.9798,'sr':2.5332,'sv-SE':2.9709,
        'sw':3.0637,'ta':2.2624,'te':2.1188,'tg':2.4971,'th':2.5947,
        'tk':2.9572,'tr':2.8806,'tt':2.5280,'uk':2.5012,'ur':2.4332,
        'uz':3.1077,'vi':2.5936,'yi':2.3051,'yo':2.5141,'zh-CN':2.6759,
        'zh-HK':2.6507,'zh-TW':2.7535,
        '__avg__': 2.7267,
    },
    # ── Chatterbox ────────────────────────────────────────────────────────────
    {
        'ar':2.7880,'da':2.8204,'de':3.0291,'el':2.7216,'en':3.2585,
        'es':2.8846,'fi':2.7206,'fr':2.9151,'he':2.8807,'hi':2.8223,
        'it':2.8648,'ja':2.7015,'ko':2.8574,'nl':3.0348,'nn-NO':2.8826,
        'pl':2.7936,'pt':2.8441,'ru':2.8646,'sv-SE':2.6231,'sw':2.9911,
        'tr':2.7691,'zh-CN':2.6391,'zh-TW':2.6253,
        '__avg__': 2.8405,
    },
    # ── Fish Audio S2 Pro ─────────────────────────────────────────────────────
    {
        'af':3.0865,'am':2.7832,'ar':2.9378,'as':2.9876,'az':3.0641,
        'ba':2.7391,'be':3.0450,'bg':2.9889,'bn':3.0289,'br':2.8960,
        'ca':3.0211,'cs':3.0426,'cy':3.0701,'da':3.0309,'de':3.2279,
        'el':2.9599,'en':3.4690,'es':3.0896,'et':3.1152,'eu':3.0561,
        'fa':2.9902,'fi':3.1364,'fr':3.1376,'gl':3.1084,'ha':2.9785,
        'he':3.0091,'hi':3.0060,'ht':2.8443,'hu':3.1362,'hy-AM':2.9400,
        'id':3.0981,'is':3.1808,'it':3.0660,'ja':2.7074,'ka':3.0342,
        'kk':2.8745,'ko':2.9186,'lo':2.6337,'lt':3.1242,'lv':3.0345,
        'mk':3.0280,'ml':2.5700,'mn':2.9345,'mr':2.8192,'mt':2.9729,
        'ne-NP':2.7929,'nl':3.1947,'nn-NO':3.0548,'oc':2.9600,'pa-IN':2.5228,
        'pl':3.0836,'ps':2.8921,'pt':3.0262,'ro':3.0390,'ru':3.0412,
        'sd':2.8330,'sk':2.9298,'sl':2.9982,'sq':2.9999,'sr':2.8833,
        'sv-SE':2.9870,'sw':2.9219,'ta':2.7453,'te':2.5483,'tg':2.9765,
        'th':2.8463,'tk':2.9744,'tr':2.8964,'tt':2.8753,'uk':2.9634,
        'ur':3.0278,'uz':3.0313,'vi':2.8471,'yi':2.9735,'yo':2.8477,
        'zh-CN':3.0697,'zh-HK':2.9997,'zh-TW':2.9403,
        '__avg__': 2.9698,
    },
    # ── Qwen3 TTS ─────────────────────────────────────────────────────────────
    {
        'de':2.7600,'en':3.0221,'es':2.6931,'fr':2.5720,'it':2.4546,
        'ja':2.5274,'ko':2.4355,'pt':2.5732,'ru':2.3231,
        'zh-CN':2.7834,'zh-TW':2.5358,
        '__avg__': 2.6073,
    },
]

# ══════════════════════════════════════════════════════════════════════════════
#  METRICS  ── list of tables to render top → bottom
# ══════════════════════════════════════════════════════════════════════════════
METRICS = [
    dict(
        data            = MOS_DATA,
        title           = 'Mean Opinion Score (UTMOSv2 5 repetitions)',
        subtitle        = '↑  higher is better',
        cmap_colors     = ['#922b21', '#e67e22', '#f9e79f',
                           '#1e8449', '#1a5276'],
        vmin            = 1.00,
        vmax            = 5.00,
        higher_is_better= True,
    ),
    dict(
        data            = CER_DATA,
        title           = 'Character Error Rate (whisper-large-v3)',
        subtitle        = '↓  lower is better',
        cmap_colors     = ['#1a5276', '#1e8449', '#f9e79f',
                           '#e67e22', '#922b21'],
        vmin            = 0.00,
        vmax            = 1.00,
        higher_is_better= False,
    ),
]

# ══════════════════════════════════════════════════════════════════════════════
#  STYLE  ── tweak font sizes, cell dimensions, colors here
# ══════════════════════════════════════════════════════════════════════════════
STYLE = dict(
    bg_color        = '#ffffff',   # figure / axes background
    cell_value_size = 5.2,         # font size inside each cell
    avg_value_size  = 5.8,         # font size in AVG column
    lang_label_size = 6.2,         # language code on top
    model_label_size= 7.5,         # model name on the left
    title_size      = 13,          # per-table title
    main_title_size = 17,          # global title at the top
    caption_size    = 8.5,         # subtitle below global title
    lang_color      = '#444444',
    model_color     = '#222222',
    avg_label_color = '#203882',
    title_color     = '#1a1a2e',
    caption_color   = '#666666',
    missing_bg      = '#eeeeee',   # cell bg when data is missing
    missing_text    = '#aaaaaa',   # cell text when data is missing
    cell_w          = 0.55,        # inches per language column
    cell_h          = 0.55,        # inches per model row
    avg_col_w       = 0.85,        # inches for the AVG column
    model_label_w   = 2.2,         # inches reserved for model names
    colorbar_w      = 0.35,        # inches for each colorbar
    gap_between     = 0.6,         # inches between tables
    margin_l        = 0.15,        # left margin
    margin_b        = 0.35,        # bottom margin
    dpi             = 300,
    output_file     = 'benchmark_results.png',
)

# ══════════════════════════════════════════════════════════════════════════════
#  RENDERING  (no need to edit below unless changing layout logic)
# ══════════════════════════════════════════════════════════════════════════════

def build_matrix(data_rows, langs):
    """Convert list-of-dicts to numpy matrix; missing keys → NaN."""
    mat = np.full((len(data_rows), len(langs)), np.nan)
    for i, row in enumerate(data_rows):
        for j, lang in enumerate(langs):
            if lang in row:
                mat[i, j] = row[lang]
    return mat


def make_cmap(colors, n=256):
    return LinearSegmentedColormap.from_list('custom', colors, N=n)


def draw_table(fig, ax_heat, ax_avg, ax_cb,
               matrix, avgs, cmap, vmin, vmax,
               title, subtitle, models, langs, s):
    """Render one heatmap table onto pre-created axes."""

    n_rows, n_cols = matrix.shape

    # ── cells ─────────────────────────────────────────────────────────────────
    for r in range(n_rows):
        for c in range(n_cols):
            v = matrix[r, c]
            row_y = n_rows - r - 1   # flip so row 0 is at top

            if np.isnan(v):
                fc   = s['missing_bg']
                txt  = '–'
                tc   = s['missing_text']
            else:
                norm = np.clip((v - vmin) / (vmax - vmin), 0, 1)
                fc   = cmap(norm)
                txt  = f'{v:.3f}'
                lum  = 0.299*fc[0] + 0.587*fc[1] + 0.114*fc[2]
                tc   = '#0a0f14' if lum > 0.45 else '#e8f4f8'

            ax_heat.add_patch(
                plt.Rectangle([c, row_y], 1, 1,
                              facecolor=fc, edgecolor=s['bg_color'],
                              linewidth=0.4))
            ax_heat.text(c + 0.5, row_y + 0.5, txt,
                         ha='center', va='center',
                         fontsize=s['cell_value_size'],
                         color=tc, fontweight='bold',
                         fontfamily='monospace')

    # ── AVG column ────────────────────────────────────────────────────────────
    for r in range(n_rows):
        v     = avgs[r]
        row_y = n_rows - r - 1
        if np.isnan(v):
            fc  = s['missing_bg']
            tc  = s['missing_text']
            txt = '–'
        else:
            norm = np.clip((v - vmin) / (vmax - vmin), 0, 1)
            fc   = cmap(norm)
            lum  = 0.299*fc[0] + 0.587*fc[1] + 0.114*fc[2]
            tc   = '#0a0f14' if lum > 0.45 else '#e8f4f8'
            txt  = f'{v:.4f}'
        ax_avg.add_patch(
            plt.Rectangle([0, row_y], 1, 1,
                          facecolor=fc, edgecolor=s['bg_color'],
                          linewidth=0.6))
        ax_avg.text(0.5, row_y + 0.5, txt,
                    ha='center', va='center',
                    fontsize=s['avg_value_size'],
                    color=tc, fontweight='bold',
                    fontfamily='monospace')

    # ── axes limits & ticks ───────────────────────────────────────────────────
    ax_heat.set_xlim(0, n_cols)
    ax_heat.set_ylim(0, n_rows)

    # language labels (top)
    ax_heat.set_xticks(np.arange(n_cols) + 0.5)
    ax_heat.set_xticklabels(langs, rotation=60, ha='right',
                             fontsize=s['lang_label_size'],
                             color=s['lang_color'], fontfamily='monospace')
    ax_heat.tick_params(axis='x', length=0, pad=2)
    ax_heat.xaxis.set_ticks_position('top')
    ax_heat.xaxis.set_label_position('top')

    # model labels (left)
    ax_heat.set_yticks(np.arange(n_rows) + 0.5)
    ax_heat.set_yticklabels(list(reversed(models)),
                             fontsize=s['model_label_size'],
                             color=s['model_color'], fontweight='bold')
    ax_heat.tick_params(axis='y', length=0, pad=6)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    # AVG column ticks
    ax_avg.set_xlim(0, 1)
    ax_avg.set_ylim(0, n_rows)
    ax_avg.set_xticks([0.5])
    ax_avg.set_xticklabels(['AVG'], fontsize=s['avg_value_size'] + 1,
                            color=s['avg_label_color'], fontweight='bold',
                            fontfamily='monospace')
    ax_avg.xaxis.set_ticks_position('top')
    ax_avg.xaxis.set_label_position('top')
    ax_avg.tick_params(axis='x', length=0, pad=2)
    ax_avg.set_yticks([])
    for spine in ax_avg.spines.values():
        spine.set_visible(False)
    ax_avg.axvline(0, color=s['avg_label_color'], linewidth=1.5)

    # ── title ─────────────────────────────────────────────────────────────────
    ax_heat.set_title(f'{title}   {subtitle}',
                      fontsize=s['title_size'],
                      color=s['title_color'], fontweight='bold',
                      pad=30, loc='left')

    # ── colorbar ──────────────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cb, orientation='vertical')
    cb.ax.yaxis.set_tick_params(color=s['lang_color'],
                                labelsize=s['lang_label_size'])
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=s['lang_color'])
    cb.outline.set_edgecolor('#cccccc')
    cb.ax.set_facecolor(s['bg_color'])


def main():
    s = STYLE
    n_models = len(MODELS)
    n_langs  = len(LANGS)
    n_metrics = len(METRICS)

    table_w = s['model_label_w'] + n_langs * s['cell_w'] + s['avg_col_w']
    table_h = 0.5 + n_models * s['cell_h']   # 0.5 = header row

    total_w = table_w + s['colorbar_w'] + 0.8
    total_h = (1.2                              # global title
               + table_h * n_metrics
               + s['gap_between'] * (n_metrics - 1)
               + 1.4)                           # bottom margin

    fig = plt.figure(figsize=(total_w, total_h), dpi=s['dpi'])
    fig.patch.set_facecolor(s['bg_color'])

    fw, fh = total_w, total_h

    def to_frac(x, y, w, h):
        return x/fw, y/fh, w/fw, h/fh

    heat_w_in = n_langs * s['cell_w']
    avg_w_in  = s['avg_col_w']
    cb_left   = s['margin_l'] + s['model_label_w'] + heat_w_in + avg_w_in + 0.12

    # draw tables from top to bottom
    for idx, metric in enumerate(METRICS):
        t_bottom = (s['margin_b']
                    + (n_metrics - 1 - idx) * (table_h + s['gap_between']))

        matrix = build_matrix(metric['data'], LANGS)
        avgs   = [row['__avg__'] for row in metric['data']]
        cmap   = make_cmap(metric['cmap_colors'])

        heat_left = s['margin_l'] + s['model_label_w']

        ax_h = fig.add_axes(to_frac(heat_left,             t_bottom, heat_w_in, table_h))
        ax_a = fig.add_axes(to_frac(heat_left + heat_w_in, t_bottom, avg_w_in,  table_h))
        ax_c = fig.add_axes(to_frac(cb_left, t_bottom + 0.3, s['colorbar_w'], table_h - 0.5))

        for ax in (ax_h, ax_a, ax_c):
            ax.set_facecolor(s['bg_color'])

        draw_table(fig, ax_h, ax_a, ax_c,
                   matrix, avgs, cmap,
                   metric['vmin'], metric['vmax'],
                   metric['title'], metric['subtitle'],
                   MODELS, LANGS, s)

    # ── global title ──────────────────────────────────────────────────────────
    top_y = s['margin_b'] + table_h * n_metrics + s['gap_between'] * (n_metrics - 1)
    fig.text(0.5, (top_y + 0.80) / fh,
             'Multilingual TTS — Model Benchmark  (76 languages)',
             ha='center', va='bottom',
             fontsize=s['main_title_size'],
             color=s['title_color'], fontweight='bold')

    fig.text(0.5, (top_y + 0.55) / fh,
             'MOS: mean opinion score (UTMOSv2, 1–5 scale)  •  '
             'CER: character error rate from ASR transcription',
             ha='center', va='bottom',
             fontsize=s['caption_size'],
             color=s['caption_color'], style='italic')

    # ── save ──────────────────────────────────────────────────────────────────
    out = s['output_file']
    plt.savefig(out, dpi=s['dpi'], bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    print(f'Saved → {out}')
    print(f'Size  : {total_w:.1f} × {total_h:.1f} in  @ {s["dpi"]} dpi')
    print(f'Pixels: {int(total_w*s["dpi"])} × {int(total_h*s["dpi"])}')


# ══════════════════════════════════════════════════════════════════════════════
#  SCATTER PLOT — CER vs Parameter Size
#  Edit MODEL_POINTS to update scores or add new models.
#  scicom=True  → highlighted as Scicom model (teal, connected by line)
#  langs        → number of evaluated languages (shown in annotation)
#  note         → optional extra annotation text
# ══════════════════════════════════════════════════════════════════════════════

MODEL_POINTS = [
    dict(label='Dia TTS',                         params=1.6, cer=0.8131, mos=1.8575, scicom=False, langs=76),
    dict(label='Multilingual-Expressive-TTS-0.6B', params=0.6, cer=0.2384, mos=3.2273, scicom=True,  langs=76),
    dict(label='Multilingual-Expressive-TTS-1.7B', params=1.7, cer=0.2362, mos=3.2330, scicom=True,  langs=76),
    dict(label='Orpheus',                         params=3.0, cer=0.6075, mos=2.7267, scicom=False, langs=76),
    dict(label='Fish Audio S2 Pro',               params=5.0, cer=0.2370, mos=2.9698, scicom=False, langs=76),
]

SCATTER_STYLE = dict(
    bg_color        = '#ffffff',
    grid_color      = '#e0e0e0',
    scicom_color    = '#203882',   # Scicom models
    scicom_edge     = '#0f1f5c',
    other_color     = '#27ae60',   # green — other models
    other_edge      = '#1a8a48',
    line_color      = '#203882',   # line connecting Scicom models
    title_color     = '#1a1a2e',
    label_color     = '#333333',
    tick_color      = '#444444',
    caption_color   = '#666666',
    anno_bg         = '#f5f5f5',
    region_color    = '#203882',   # shaded "small model" region
    marker_size     = 220,         # scatter dot area
    scicom_size     = 280,
    label_fontsize  = 9.5,
    title_fontsize  = 13,
    axis_fontsize   = 10,
    tick_fontsize   = 9,
    dpi             = 300,
    output_file     = 'scatter_results.png',
)


def draw_scatter(ax, points, y_key, y_label, y_lim, title, ss, label_offsets=None):
    """Draw one scatter panel. Points with None for y_key are skipped."""

    active = [p for p in points if p.get(y_key) is not None]

    ax.set_facecolor(ss['bg_color'])
    for spine in ax.spines.values():
        spine.set_color('#cccccc')

    # shaded "small model" region (< 2B params)
    ax.axvspan(0, 2.0, color=ss['region_color'], alpha=0.06, zorder=0)
    ax.text(1.0, y_lim[0] + (y_lim[1] - y_lim[0]) * 0.02,
            '< 2B region', ha='center', va='bottom',
            fontsize=8, color=ss['region_color'], alpha=0.5, style='italic')

    ax.grid(True, color=ss['grid_color'], linewidth=0.8,
            linestyle='--', alpha=0.6, zorder=1)
    ax.set_axisbelow(True)

    # dashed line connecting Scicom models (sorted by params)
    scicom_pts = sorted([p for p in active if p['scicom']], key=lambda p: p['params'])
    if scicom_pts:
        ax.plot([p['params'] for p in scicom_pts],
                [p[y_key]   for p in scicom_pts],
                color=ss['line_color'], linewidth=1.8,
                linestyle='--', alpha=0.55, zorder=2)

    # dots
    for p in active:
        x, y = p['params'], p[y_key]
        if p['scicom']:
            color, edge, size, lw, zo = (ss['scicom_color'], ss['scicom_edge'],
                                         ss['scicom_size'], 2.0, 5)
        else:
            color, edge, size, lw, zo = (ss['other_color'], ss['other_edge'],
                                         ss['marker_size'], 1.2, 4)
        ax.scatter(x, y, s=size, color=color, edgecolors=edge,
                   linewidths=lw, zorder=zo)

    # label annotations — nudge positions to avoid overlap
    default_offsets = {
        'Dia TTS':               ( 0.12, -0.070),
        'Multilingual-Expressive-TTS-0.6B':( 0.12,  0.060),
        'Multilingual-Expressive-TTS-1.7B':( 0.15,  0.020),
        'Orpheus':               ( 0.12,  0.020),
        'Fish Audio S2 Pro':     (-0.45,  0.060),
        'Qwen3 TTS':             (-1.20, -0.070),
    }
    if label_offsets:
        default_offsets.update(label_offsets)
    for p in active:
        x, y    = p['params'], p[y_key]
        dx, dy  = default_offsets.get(p['label'], (0.12, 0.020))
        color   = ss['scicom_color'] if p['scicom'] else ss['label_color']
        weight  = 'bold' if p['scicom'] else 'normal'
        note      = f" ({p.get('note')})" if p.get('note') else ''
        param_str = f"{p['params']}B"
        score_txt = f"{y:.4f}  ·  {param_str}{note}"

        ax.annotate(
            f"{p['label'].replace(chr(10), ' ')}",
            xy=(x, y), xytext=(x + dx, y + dy),
            fontsize=ss['label_fontsize'],
            color=color, fontweight=weight,
            arrowprops=None,
            bbox=dict(boxstyle='round,pad=0.25', fc=ss['anno_bg'], ec='none', alpha=0.75),
        )
        ax.text(x + dx, y + dy - (y_lim[1] - y_lim[0]) * 0.058,
                score_txt, fontsize=7.8, color=color, alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.15', fc=ss['anno_bg'], ec='none', alpha=0.6))

    # arrow indicator (top-right corner)
    arrow = '↑ higher is better' if '↑' in title else '↓ lower is better'
    arrow_color = '#00c9a7' if '↑' in title else '#f4845f'
    ax.text(0.98, 0.97, arrow,
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, color=arrow_color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc=ss['anno_bg'], ec=arrow_color,
                      alpha=0.85, linewidth=1.2))

    ax.set_xlim(-0.1, 5.8)
    ax.set_ylim(*y_lim)

    ax.set_xlabel('Parameter Size (B)', fontsize=ss['axis_fontsize'],
                  color=ss['tick_color'], labelpad=6)
    ax.set_ylabel(y_label, fontsize=ss['axis_fontsize'],
                  color=ss['tick_color'], labelpad=6)
    ax.tick_params(colors=ss['tick_color'], labelsize=ss['tick_fontsize'])
    ax.set_title(title, fontsize=ss['title_fontsize'],
                 color=ss['title_color'], fontweight='bold', pad=10)

    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xticklabels(['0B', '1B', '2B', '3B', '4B', '5B'],
                       color=ss['tick_color'], fontsize=ss['tick_fontsize'])


def scatter_main():
    ss = SCATTER_STYLE
    fig, (ax_mos, ax_cer) = plt.subplots(1, 2, figsize=(14, 6), dpi=ss['dpi'])
    fig.patch.set_facecolor(ss['bg_color'])
    fig.subplots_adjust(wspace=0.32, left=0.07, right=0.97,
                        top=0.86, bottom=0.12)

    draw_scatter(ax_mos, MODEL_POINTS,
                 y_key='mos',
                 y_label='MOS (UTMOSv2 5 repetitions)',
                 y_lim=(1.5, 3.8),
                 title='MOS ↑  vs Parameter Size',
                 ss=ss,
                 label_offsets={
                     'Multilingual-Expressive-TTS-0.6B': ( 0.12,  0.055),
                     'Multilingual-Expressive-TTS-1.7B': ( 0.12, -0.100),
                     'Fish Audio S2 Pro':                (-0.45,  0.055),
                 })

    draw_scatter(ax_cer, MODEL_POINTS,
                 y_key='cer',
                 y_label='Character Error Rate (whisper-large-v3)',
                 y_lim=(-0.02, 0.95),
                 title='CER ↓  vs Parameter Size',
                 ss=ss)

    # ── legend ────────────────────────────────────────────────────────────────
    import matplotlib.lines as mlines
    scicom_dot  = mlines.Line2D([], [], marker='o', linestyle='None',
                                color=ss['scicom_color'],
                                markeredgecolor=ss['scicom_edge'],
                                markeredgewidth=1.5,
                                markersize=10, label='Scicom (ours)')
    other_dot   = mlines.Line2D([], [], marker='o', linestyle='None',
                                color=ss['other_color'],
                                markeredgecolor=ss['other_edge'],
                                markeredgewidth=1.2,
                                markersize=9,  label='Other models')
    scicom_line = mlines.Line2D([], [], color=ss['line_color'],
                                linewidth=1.8, linestyle='--',
                                label='Scicom scaling curve')
    region_patch = plt.Rectangle((0,0),1,1, fc=ss['region_color'],
                                  alpha=0.15, label='< 2B param region')

    for ax in (ax_mos, ax_cer):
        ax.legend(handles=[scicom_dot, other_dot, scicom_line, region_patch],
                  fontsize=8, facecolor='#f5f5f5', edgecolor='#cccccc',
                  labelcolor=ss['label_color'],
                  loc='upper left', framealpha=0.95)

    # ── global title ──────────────────────────────────────────────────────────
    fig.suptitle(
        'Score vs Parameter Size  —  Scicom Models Are Competitive at Small Scale',
        fontsize=13.5, color=ss['title_color'], fontweight='bold', y=0.97)
    fig.text(0.5, 0.915,
             'Scicom Multilingual TTS achieves strong performance with fewer parameters',
             ha='center', fontsize=8.5,
             color=ss['caption_color'], style='italic')

    out = ss['output_file']
    plt.savefig(out, dpi=ss['dpi'], bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()
    scatter_main()
