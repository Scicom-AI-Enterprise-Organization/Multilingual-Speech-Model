"""
plot_results.py
───────────────
Generates a large PNG heatmap comparing TTS/VC models across languages.

To add a new model:   add an entry to MODELS and a row to SIM_DATA / CER_DATA.
To add a new metric:  add a new section in the METRICS list at the bottom.
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
#  MODELS  (row order — edit display name and optional short tag)
# ══════════════════════════════════════════════════════════════════════════════
MODELS = [
    'Dia TTS',
    'Multilingual TTS 0.6B',
    'Multilingual TTS 1.7B',
    'Orpheus',
    'Chatterbox',
    'Fish Audio S2 Pro',
]

# ══════════════════════════════════════════════════════════════════════════════
#  DATA  ── each row is a dict {lang_code: score}
#          missing keys → shown as "–" in the plot
#          language codes must match entries in LANGS exactly
# ══════════════════════════════════════════════════════════════════════════════

SIM_DATA = [
    # ── Dia TTS ──────────────────────────────────────────────────────────────
    {
        'af':0.3337,'am':0.2528,'ar':0.2600,'as':0.3283,'az':0.4802,
        'ba':0.2114,'be':0.3000,'bg':0.1899,'bn':0.3387,'br':0.2780,
        'ca':0.4604,'cs':0.4725,'cy':0.3492,'da':0.4131,'de':0.4162,
        'el':0.1883,'en':0.3945,'es':0.5141,'et':0.5140,'eu':0.5320,
        'fa':0.2358,'fi':0.4246,'fr':0.4552,'gl':0.4735,'ha':0.4259,
        'he':0.2861,'hi':0.2299,'ht':0.3286,'hu':0.4934,'hy-AM':0.2462,
        'id':0.4291,'is':0.5780,'it':0.4742,'ja':0.2375,'ka':0.3056,
        'kk':0.2684,'ko':0.2775,'lo':0.4145,'lt':0.5183,'lv':0.4272,
        'mk':0.1870,'ml':0.2286,'mn':0.2007,'mr':0.3199,'mt':0.4118,
        'ne-NP':0.2240,'nl':0.3822,'nn-NO':0.3982,'oc':0.3970,'pa-IN':0.2699,
        'pl':0.4170,'ps':0.2219,'pt':0.4123,'ro':0.4659,'ru':0.1558,
        'sd':0.2208,'sk':0.4039,'sl':0.3981,'sq':0.4697,'sr':0.1547,
        'sv-SE':0.3632,'sw':0.5118,'ta':0.3124,'te':0.2680,'tg':0.1748,
        'th':0.2938,'tk':0.4787,'tr':0.3589,'tt':0.2493,'uk':0.1287,
        'ur':0.2346,'uz':0.4509,'vi':0.3326,'yi':0.2193,'yo':0.4864,
        'zh-CN':0.2815,'zh-HK':0.2585,'zh-TW':0.1461,
        '__avg__': 0.3416,
    },
    # ── Multilingual TTS 0.6B ────────────────────────────────────────────────
    {
        'af':0.5121,'am':0.5132,'ar':0.4300,'as':0.5331,'az':0.5316,
        'ba':0.4551,'be':0.5979,'bg':0.6095,'bn':0.5832,'br':0.3314,
        'ca':0.5070,'cs':0.4890,'cy':0.4149,'da':0.4135,'de':0.5412,
        'el':0.4422,'en':0.5087,'es':0.5552,'et':0.5698,'eu':0.5635,
        'fa':0.4468,'fi':0.4590,'fr':0.5369,'gl':0.5093,'ha':0.4805,
        'he':0.4929,'hi':0.4921,'ht':0.3976,'hu':0.5103,'hy-AM':0.5831,
        'id':0.4808,'is':0.5248,'it':0.5697,'ja':0.4444,'ka':0.5860,
        'kk':0.4935,'ko':0.5508,'lo':0.4723,'lt':0.5480,'lv':0.4431,
        'mk':0.5880,'ml':0.4207,'mn':0.5209,'mr':0.5802,'mt':0.4761,
        'ne-NP':0.4764,'nl':0.5041,'nn-NO':0.4268,'oc':0.4497,'pa-IN':0.5042,
        'pl':0.4855,'ps':0.4846,'pt':0.4039,'ro':0.5348,'ru':0.5169,
        'sd':0.4272,'sk':0.4075,'sl':0.4358,'sq':0.4804,'sr':0.3240,
        'sv-SE':0.3691,'sw':0.5102,'ta':0.5260,'te':0.4663,'tg':0.5160,
        'th':0.4613,'tk':0.4945,'tr':0.3735,'tt':0.4430,'uk':0.4826,
        'ur':0.5032,'uz':0.5294,'vi':0.3593,'yi':0.4014,'yo':0.5081,
        'zh-CN':0.5086,'zh-HK':0.5203,'zh-TW':0.4256,
        '__avg__': 0.4868,
    },
    # ── Multilingual TTS 1.7B ────────────────────────────────────────────────
    {
        'af':0.5300,'am':0.5480,'ar':0.4484,'as':0.5358,'az':0.5675,
        'ba':0.4666,'be':0.6024,'bg':0.6113,'bn':0.5791,'br':0.3648,
        'ca':0.5173,'cs':0.5014,'cy':0.4528,'da':0.4369,'de':0.5347,
        'el':0.4589,'en':0.5124,'es':0.5678,'et':0.5877,'eu':0.5648,
        'fa':0.4871,'fi':0.4907,'fr':0.5380,'gl':0.5273,'ha':0.5105,
        'he':0.4942,'hi':0.4980,'ht':0.4492,'hu':0.5218,'hy-AM':0.5819,
        'id':0.4866,'is':0.5673,'it':0.5737,'ja':0.4527,'ka':0.5821,
        'kk':0.5123,'ko':0.5484,'lo':0.5993,'lt':0.5594,'lv':0.4847,
        'mk':0.5926,'ml':0.4237,'mn':0.5453,'mr':0.5886,'mt':0.4936,
        'ne-NP':0.4876,'nl':0.5031,'nn-NO':0.4550,'oc':0.4516,'pa-IN':0.5041,
        'pl':0.4995,'ps':0.5196,'pt':0.4211,'ro':0.5545,'ru':0.5239,
        'sd':0.4301,'sk':0.4262,'sl':0.4579,'sq':0.4986,'sr':0.3531,
        'sv-SE':0.3984,'sw':0.5222,'ta':0.5216,'te':0.4552,'tg':0.5320,
        'th':0.4764,'tk':0.5322,'tr':0.3903,'tt':0.4600,'uk':0.4999,
        'ur':0.5162,'uz':0.5449,'vi':0.3810,'yi':0.4397,'yo':0.5272,
        'zh-CN':0.5115,'zh-HK':0.5441,'zh-TW':0.4428,
        '__avg__': 0.5036,
    },
    # ── Orpheus ──────────────────────────────────────────────────────────────
    {
        'af':0.4950,'am':0.3663,'ar':0.2850,'as':0.3605,'az':0.4466,
        'ba':0.3195,'be':0.4125,'bg':0.4543,'bn':0.3445,'br':0.3272,
        'ca':0.4816,'cs':0.4084,'cy':0.3938,'da':0.3953,'de':0.4978,
        'el':0.3316,'en':0.4685,'es':0.5430,'et':0.5518,'eu':0.5390,
        'fa':0.3276,'fi':0.4352,'fr':0.4896,'gl':0.4804,'ha':0.4642,
        'he':0.3426,'hi':0.3370,'ht':0.3526,'hu':0.4881,'hy-AM':0.3906,
        'id':0.4497,'is':0.5186,'it':0.5479,'ja':0.3311,'ka':0.4027,
        'kk':0.3570,'ko':0.4240,'lo':0.3593,'lt':0.5255,'lv':0.4343,
        'mk':0.4379,'ml':0.2859,'mn':0.3823,'mr':0.3434,'mt':0.4367,
        'ne-NP':0.3355,'nl':0.4856,'nn-NO':0.4093,'oc':0.4280,'pa-IN':0.3826,
        'pl':0.4314,'ps':0.3360,'pt':0.3854,'ro':0.5037,'ru':0.3842,
        'sd':0.2976,'sk':0.3698,'sl':0.4274,'sq':0.4471,'sr':0.2596,
        'sv-SE':0.3503,'sw':0.4772,'ta':0.3173,'te':0.3177,'tg':0.3805,
        'th':0.3184,'tk':0.4201,'tr':0.3165,'tt':0.3313,'uk':0.3516,
        'ur':0.3167,'uz':0.5045,'vi':0.2812,'yi':0.2832,'yo':0.3770,
        'zh-CN':0.4245,'zh-HK':0.4177,'zh-TW':0.3775,
        '__avg__': 0.4002,
    },
    # ── Chatterbox  (only 23 languages) ─────────────────────────────────────
    {
        'ar':0.6326,'da':0.6584,'de':0.7088,'el':0.6365,'en':0.6579,
        'es':0.7312,'fi':0.6916,'fr':0.7068,'he':0.6797,'hi':0.6675,
        'it':0.7334,'ja':0.6203,'ko':0.7314,'nl':0.7002,'nn-NO':0.6669,
        'pl':0.6741,'pt':0.6048,'ru':0.6912,'sv-SE':0.6264,'sw':0.7075,
        'tr':0.5785,'zh-CN':0.6864,'zh-TW':0.6282,
        '__avg__': 0.6704,
    },
    # ── Fish Audio S2 Pro ──────────────────────────────────────────────────────────
    {
        'af':0.6303,'am':0.6420,'ar':0.5493,'as':0.6231,'az':0.6362,
        'ba':0.5654,'be':0.7089,'bg':0.6922,'bn':0.6682,'br':0.4680,
        'ca':0.6552,'cs':0.6141,'cy':0.5657,'da':0.5762,'de':0.6543,
        'el':0.5659,'en':0.6184,'es':0.6822,'et':0.7121,'eu':0.7001,
        'fa':0.5899,'fi':0.6106,'fr':0.6515,'gl':0.6422,'ha':0.6225,
        'he':0.6428,'hi':0.5928,'ht':0.5075,'hu':0.6535,'hy-AM':0.6949,
        'id':0.5902,'is':0.6913,'it':0.6863,'ja':0.5344,'ka':0.6940,
        'kk':0.6091,'ko':0.6711,'lo':0.6909,'lt':0.6793,'lv':0.5903,
        'mk':0.6877,'ml':0.5262,'mn':0.6382,'mr':0.6842,'mt':0.6153,
        'ne-NP':0.5912,'nl':0.6451,'nn-NO':0.5781,'oc':0.5797,'pa-IN':0.6085,
        'pl':0.6036,'ps':0.6178,'pt':0.5269,'ro':0.6521,'ru':0.6240,
        'sd':0.5578,'sk':0.5187,'sl':0.5573,'sq':0.5965,'sr':0.4204,
        'sv-SE':0.5239,'sw':0.6402,'ta':0.6561,'te':0.5545,'tg':0.6117,
        'th':0.5817,'tk':0.6155,'tr':0.4785,'tt':0.5614,'uk':0.5979,
        'ur':0.6036,'uz':0.6340,'vi':0.4726,'yi':0.5536,'yo':0.6621,
        'zh-CN':0.6185,'zh-HK':0.6413,'zh-TW':0.5470,
        '__avg__': 0.6097,
    },
]

CER_DATA = [
    # ── Dia TTS ──────────────────────────────────────────────────────────────
    {
        'af':0.3029,'am':0.9998,'ar':0.9533,'as':0.9901,'az':0.5508,
        'ba':0.9895,'be':0.9419,'bg':0.9449,'bn':0.9789,'br':0.6220,
        'ca':0.2025,'cs':0.4114,'cy':0.4794,'da':0.4549,'de':0.2062,
        'el':0.9690,'en':0.1707,'es':0.1019,'et':0.3007,'eu':0.1829,
        'fa':0.9719,'fi':0.3899,'fr':0.2443,'gl':0.2058,'ha':0.3496,
        'he':0.9576,'hi':0.9757,'ht':0.5741,'hu':0.3764,'hy-AM':0.9565,
        'id':0.3050,'is':0.3931,'it':0.1502,'ja':0.9863,'ka':0.9712,
        'kk':0.9796,'ko':0.9720,'lo':0.9972,'lt':0.3816,'lv':0.5298,
        'mk':0.9514,'ml':0.9919,'mn':0.9845,'mr':0.9645,'mt':0.4463,
        'ne-NP':0.9882,'nl':0.1934,'nn-NO':0.3921,'oc':0.4540,'pa-IN':0.9960,
        'pl':0.5089,'ps':0.9634,'pt':0.4754,'ro':0.3127,'ru':0.9377,
        'sd':0.9891,'sk':0.7994,'sl':0.4282,'sq':0.4066,'sr':0.9889,
        'sv-SE':0.4409,'sw':0.2750,'ta':0.9731,'te':0.9972,'tg':0.9800,
        'th':0.9910,'tk':0.5880,'tr':0.6333,'tt':0.9784,'uk':0.9571,
        'ur':0.9531,'uz':0.4078,'vi':0.9794,'yi':0.9791,'yo':0.8336,
        'zh-CN':0.9979,'zh-HK':0.9997,'zh-TW':0.9998,
        '__avg__': 0.6867,
    },
    # ── Multilingual TTS 0.6B ────────────────────────────────────────────────
    {
        'af':0.2587,'am':1.0000,'ar':0.2720,'as':0.9216,'az':0.2665,
        'ba':0.8456,'be':0.1466,'bg':0.1593,'bn':0.2635,'br':0.5497,
        'ca':0.1982,'cs':0.2321,'cy':0.5757,'da':0.3154,'de':0.0330,
        'el':0.2938,'en':0.0657,'es':0.0966,'et':0.2390,'eu':0.1573,
        'fa':0.3428,'fi':0.2762,'fr':0.0705,'gl':0.1743,'ha':0.3236,
        'he':0.3846,'hi':0.1595,'ht':0.3943,'hu':0.2915,'hy-AM':0.1321,
        'id':0.0632,'is':0.4129,'it':0.1390,'ja':0.2748,'ka':0.1377,
        'kk':0.2805,'ko':0.0816,'lo':0.9996,'lt':0.2221,'lv':0.3340,
        'mk':0.1070,'ml':0.9664,'mn':0.4968,'mr':0.2113,'mt':0.4565,
        'ne-NP':0.3428,'nl':0.1234,'nn-NO':0.3561,'oc':0.3979,'pa-IN':0.4125,
        'pl':0.2804,'ps':0.4784,'pt':0.3042,'ro':0.1627,'ru':0.1057,
        'sd':0.9646,'sk':0.4384,'sl':0.2490,'sq':0.3076,'sr':0.7663,
        'sv-SE':0.2865,'sw':0.2568,'ta':0.1771,'te':0.6547,'tg':0.3756,
        'th':0.3245,'tk':0.6428,'tr':0.2749,'tt':0.4033,'uk':0.1861,
        'ur':0.1311,'uz':0.3643,'vi':0.4255,'yi':0.5936,'yo':0.5849,
        'zh-CN':0.2480,'zh-HK':0.5884,'zh-TW':0.4835,
        '__avg__': 0.3502,
    },
    # ── Multilingual TTS 1.7B ────────────────────────────────────────────────
    {
        'af':0.1814,'am':1.0000,'ar':0.2174,'as':0.9171,'az':0.0687,
        'ba':0.7372,'be':0.1025,'bg':0.1011,'bn':0.2459,'br':0.4494,
        'ca':0.1209,'cs':0.1187,'cy':0.3503,'da':0.1841,'de':0.0245,
        'el':0.1650,'en':0.0413,'es':0.0471,'et':0.1255,'eu':0.1124,
        'fa':0.1796,'fi':0.1434,'fr':0.0601,'gl':0.0925,'ha':0.2532,
        'he':0.2584,'hi':0.1065,'ht':0.2963,'hu':0.1013,'hy-AM':0.1093,
        'id':0.0409,'is':0.1690,'it':0.0591,'ja':0.1875,'ka':0.1154,
        'kk':0.1408,'ko':0.0664,'lo':0.9984,'lt':0.1278,'lv':0.1803,
        'mk':0.0839,'ml':0.9640,'mn':0.3393,'mr':0.1944,'mt':0.3015,
        'ne-NP':0.2778,'nl':0.0450,'nn-NO':0.2347,'oc':0.3445,'pa-IN':0.3733,
        'pl':0.1075,'ps':0.3929,'pt':0.2051,'ro':0.0617,'ru':0.0658,
        'sd':0.9399,'sk':0.3436,'sl':0.1840,'sq':0.2062,'sr':0.7416,
        'sv-SE':0.1722,'sw':0.1670,'ta':0.1646,'te':0.5406,'tg':0.2769,
        'th':0.1512,'tk':0.5476,'tr':0.1982,'tt':0.3106,'uk':0.0993,
        'ur':0.0800,'uz':0.2965,'vi':0.2537,'yi':0.4751,'yo':0.5321,
        'zh-CN':0.2116,'zh-HK':0.4435,'zh-TW':0.3942,
        '__avg__': 0.2656,
    },
    # ── Orpheus ──────────────────────────────────────────────────────────────
    {
        'af':0.3086,'am':0.9982,'ar':0.9274,'as':0.9657,'az':0.7177,
        'ba':0.9655,'be':0.8430,'bg':0.7824,'bn':0.9459,'br':0.5855,
        'ca':0.2765,'cs':0.7404,'cy':0.6102,'da':0.4519,'de':0.1483,
        'el':0.9472,'en':0.1021,'es':0.1900,'et':0.3801,'eu':0.1949,
        'fa':0.8795,'fi':0.3708,'fr':0.3298,'gl':0.2465,'ha':0.3852,
        'he':0.8824,'hi':0.8939,'ht':0.4196,'hu':0.4544,'hy-AM':0.9008,
        'id':0.3449,'is':0.4379,'it':0.2140,'ja':0.9673,'ka':0.9234,
        'kk':0.8672,'ko':0.9239,'lo':0.9994,'lt':0.3732,'lv':0.4114,
        'mk':0.8200,'ml':0.9826,'mn':0.9270,'mr':0.9003,'mt':0.6079,
        'ne-NP':0.9144,'nl':0.2345,'nn-NO':0.4144,'oc':0.4571,'pa-IN':0.9653,
        'pl':0.6321,'ps':0.8765,'pt':0.4555,'ro':0.3581,'ru':0.8895,
        'sd':0.9601,'sk':0.7087,'sl':0.3687,'sq':0.5085,'sr':0.9196,
        'sv-SE':0.4179,'sw':0.3484,'ta':0.8962,'te':0.9858,'tg':0.8695,
        'th':0.9619,'tk':0.7387,'tr':0.7374,'tt':0.8962,'uk':0.9032,
        'ur':0.8716,'uz':0.4403,'vi':0.9291,'yi':0.8958,'yo':0.8706,
        'zh-CN':0.9475,'zh-HK':0.9697,'zh-TW':0.9227,
        '__avg__': 0.6771,
    },
    # ── Chatterbox  (only 23 languages) ─────────────────────────────────────
    {
        'ar':0.1577,'da':0.0956,'de':0.0434,'el':0.1077,'en':0.0554,
        'es':0.0318,'fi':0.0491,'fr':0.0814,'he':0.3864,'hi':0.1644,
        'it':0.0313,'ja':0.1669,'ko':0.0651,'nl':0.0153,'nn-NO':0.0943,
        'pl':0.0465,'pt':0.0933,'ru':0.0447,'sv-SE':0.0415,'sw':0.1117,
        'tr':0.0692,'zh-CN':0.2252,'zh-TW':0.3508,
        '__avg__': 0.1099,
    },
    # ── Fish Audio S2 Pro ──────────────────────────────────────────────────────────
    {
        'af':0.0937,'am':1.0000,'ar':0.1265,'as':0.9173,'az':0.0665,
        'ba':0.7211,'be':0.1029,'bg':0.0789,'bn':0.2223,'br':0.3083,
        'ca':0.0659,'cs':0.0379,'cy':0.2726,'da':0.1228,'de':0.0152,
        'el':0.1504,'en':0.0234,'es':0.0249,'et':0.0431,'eu':0.0681,
        'fa':0.1245,'fi':0.0249,'fr':0.0477,'gl':0.0533,'ha':0.2036,
        'he':0.2834,'hi':0.0898,'ht':0.2642,'hu':0.0341,'hy-AM':0.2006,
        'id':0.0297,'is':0.1725,'it':0.0276,'ja':0.1331,'ka':0.1514,
        'kk':0.1627,'ko':0.0435,'lo':0.9972,'lt':0.1292,'lv':0.0955,
        'mk':0.0731,'ml':0.9603,'mn':0.2852,'mr':0.1840,'mt':0.2645,
        'ne-NP':0.2461,'nl':0.0110,'nn-NO':0.1117,'oc':0.2359,'pa-IN':0.4070,
        'pl':0.0462,'ps':0.3557,'pt':0.0846,'ro':0.0588,'ru':0.0330,
        'sd':0.9668,'sk':0.1961,'sl':0.0442,'sq':0.1904,'sr':0.6995,
        'sv-SE':0.0381,'sw':0.1398,'ta':0.1096,'te':0.6089,'tg':0.2684,
        'th':0.1235,'tk':0.4439,'tr':0.0454,'tt':0.2617,'uk':0.0733,
        'ur':0.0939,'uz':0.2262,'vi':0.2739,'yi':0.4126,'yo':0.4952,
        'zh-CN':0.1792,'zh-HK':0.4260,'zh-TW':0.3994,
        '__avg__': 0.2283,
    },
]

# ══════════════════════════════════════════════════════════════════════════════
#  METRICS  ── list of tables to render top → bottom
#  Each entry:
#    data       : list of per-model dicts (same order as MODELS)
#    title      : section heading
#    subtitle   : shown next to title
#    cmap_colors: list of hex colors low→high
#    vmin/vmax  : colormap range (values outside are clamped)
#    higher_is_better: used only for documentation
# ══════════════════════════════════════════════════════════════════════════════
METRICS = [
    dict(
        data            = SIM_DATA,
        title           = 'Speaker Similarity',
        subtitle        = '↑  higher is better',
        cmap_colors     = ['#eef4fb', '#a8cce4', '#4a9abe',
                           '#1a7a55', '#a8d878', '#f4e840'],
        vmin            = 0.10,
        vmax            = 0.75,
        higher_is_better= True,
    ),
    dict(
        data            = CER_DATA,
        title           = 'Character Error Rate (CER)',
        subtitle        = '↓  lower is better',
        cmap_colors     = ['#eef6ee', '#a8dca8', '#f0e060',
                           '#e8883a', '#c0281a'],
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
    lang_color      = '#445566',
    model_color     = '#1a1a2e',
    avg_label_color = '#b07800',
    title_color     = '#1a1a2e',
    caption_color   = '#667788',
    missing_bg      = '#e8e8ee',   # cell bg when data is missing
    missing_text    = '#aaaacc',   # cell text when data is missing
    cell_w          = 0.55,        # inches per language column
    cell_h          = 0.55,        # inches per model row
    avg_col_w       = 0.85,        # inches for the AVG column
    model_label_w   = 2.2,         # inches reserved for model names
    colorbar_w      = 0.35,        # inches for each colorbar
    gap_between     = 1.1,         # inches between tables
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
                tc   = '#0a0f14' if lum > 0.45 else '#1a1a2e'

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
        norm  = np.clip((v - vmin) / (vmax - vmin), 0, 1)
        fc    = cmap(norm)
        lum   = 0.299*fc[0] + 0.587*fc[1] + 0.114*fc[2]
        tc    = '#0a0f14' if lum > 0.45 else '#e8f4f8'
        ax_avg.add_patch(
            plt.Rectangle([0, row_y], 1, 1,
                          facecolor=fc, edgecolor=s['bg_color'],
                          linewidth=0.6))
        ax_avg.text(0.5, row_y + 0.5, f'{v:.4f}',
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
                      pad=40, loc='left')

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
    total_h = (1.1                              # global title + caption clearance
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

        ax_h = fig.add_axes(to_frac(heat_left,           t_bottom, heat_w_in, table_h))
        ax_a = fig.add_axes(to_frac(heat_left + heat_w_in, t_bottom, avg_w_in, table_h))
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
    fig.text(0.5, (top_y + 0.70) / fh,
             'Multilingual Voice Cloning — Model Benchmark  (76 languages)',
             ha='center', va='bottom',
             fontsize=s['main_title_size'],
             color=s['title_color'], fontweight='bold')

    fig.text(0.5, (top_y + 0.40) / fh,
             'Speaker Similarity: cosine similarity of speaker embeddings  •  '
             'CER: character error rate from ASR transcription  •  Whisper large-v3',
             ha='center', va='bottom',
             fontsize=s['caption_size'],
             color=s['caption_color'], style='italic')

    # ── separator line between tables ─────────────────────────────────────────
    if n_metrics > 1:
        sep_y = (s['margin_b'] + table_h + s['gap_between'] * 0.45) / fh
        fig.add_artist(
            plt.Line2D([0.02, 0.98], [sep_y, sep_y],
                       transform=fig.transFigure,
                       color='#1e3a52', linewidth=1.2, linestyle='--'))

    # ── save ──────────────────────────────────────────────────────────────────
    out = s['output_file']
    plt.savefig(out, dpi=s['dpi'], bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    print(f'Saved → {out}')
    print(f'Size  : {total_w:.1f} × {total_h:.1f} in  @ {s["dpi"]} dpi')
    print(f'Pixels: {int(total_w*s["dpi"])} × {int(total_h*s["dpi"])}')


# ══════════════════════════════════════════════════════════════════════════════
#  SCATTER PLOT — Score vs Parameter Size
#  Edit MODEL_POINTS to update scores or add new models.
#  scicom=True  → highlighted as Scicom model (teal, connected by line)
#  langs        → number of evaluated languages (shown in annotation)
#  note         → optional extra annotation text
# ══════════════════════════════════════════════════════════════════════════════

MODEL_POINTS = [
    dict(label='Dia TTS',                params=1.6,  sim=0.3416, cer=0.6867, scicom=False, langs=76),
    dict(label='Multilingual\nTTS 0.6B', params=0.6,  sim=0.4868, cer=0.3502, scicom=True,  langs=76),
    dict(label='Multilingual\nTTS 1.7B', params=1.7,  sim=0.5036, cer=0.2656, scicom=True,  langs=76),
    dict(label='Orpheus',                params=3.0,  sim=0.4002, cer=0.6771, scicom=False, langs=76),
    dict(label='Fish Audio S2 Pro',            params=5.0,  sim=0.6097, cer=0.2283, scicom=False, langs=76),
]

SCATTER_STYLE = dict(
    bg_color        = '#ffffff',
    grid_color      = '#e0e0e0',
    scicom_color    = '#203882',   # rgb(32,56,130) — Scicom models
    scicom_edge     = '#101d55',
    other_color     = '#2e9e4f',   # green — other models
    other_edge      = '#1d7a3a',
    line_color      = '#203882',   # line connecting Scicom models
    title_color     = '#1a1a2e',
    label_color     = '#2c2c3e',
    tick_color      = '#555577',
    caption_color   = '#777799',
    anno_bg         = '#f5f5f5',
    region_color    = '#00a885',   # shaded "small model" region
    marker_size     = 220,         # scatter dot area
    scicom_size     = 280,
    label_fontsize  = 9.5,
    title_fontsize  = 13,
    axis_fontsize   = 10,
    tick_fontsize   = 9,
    dpi             = 150,
    output_file     = 'scatter_results.png',
)


def draw_scatter(ax, points, y_key, y_label, y_lim, title, ss, label_offsets=None):
    """Draw one scatter panel (similarity or CER)."""

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
    scicom_pts = sorted([p for p in points if p['scicom']], key=lambda p: p['params'])
    if scicom_pts:
        ax.plot([p['params'] for p in scicom_pts],
                [p[y_key]   for p in scicom_pts],
                color=ss['line_color'], linewidth=1.8,
                linestyle='--', alpha=0.55, zorder=2)

    # dots
    for p in points:
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
        'Dia TTS':               ( 0.12,  0.035),
        'Multilingual\nTTS 0.6B':( 0.12,  0.055),
        'Multilingual\nTTS 1.7B':( 0.12, -0.050),
        'Orpheus':               ( 0.12,  0.018),
        'Fish Audio S2 Pro':           (-1.05,  0.018),
    }
    if label_offsets:
        default_offsets.update(label_offsets)
    label_offsets = default_offsets
    for p in points:
        x, y    = p['params'], p[y_key]
        dx, dy  = label_offsets.get(p['label'], (0.12, 0.018))
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
            bbox=dict(boxstyle='round,pad=0.25', fc=ss['anno_bg'], ec='none', alpha=0.75),
        )
        ax.text(x + dx, y + dy - (y_lim[1] - y_lim[0]) * 0.058,
                score_txt, fontsize=7.8, color=color, alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.15', fc=ss['anno_bg'], ec='none', alpha=0.6))

    # arrow indicator (top-right corner)
    arrow = '↑ higher is better' if '↑' in title else '↓ lower is better'
    arrow_color = '#203882' if '↑' in title else '#e05c3a'
    ax.text(0.98, 0.97, arrow,
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, color=arrow_color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc=ss['anno_bg'], ec=arrow_color,
                      alpha=0.9, linewidth=1.2))

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
    fig, (ax_sim, ax_cer) = plt.subplots(1, 2, figsize=(14, 6), dpi=ss['dpi'])
    fig.patch.set_facecolor(ss['bg_color'])
    fig.subplots_adjust(wspace=0.32, left=0.07, right=0.97,
                        top=0.86, bottom=0.12)

    draw_scatter(ax_sim, MODEL_POINTS,
                 y_key='sim',
                 y_label='Speaker Similarity',
                 y_lim=(0.28, 0.82),
                 title='Speaker Similarity ↑  vs Parameter Size',
                 ss=ss,
                 label_offsets={
                     'Multilingual\nTTS 0.6B': (-0.45,  0.055),
                     'Multilingual\nTTS 1.7B': ( 0.08, -0.030),
                 })

    draw_scatter(ax_cer, MODEL_POINTS,
                 y_key='cer',
                 y_label='Character Error Rate (CER)',
                 y_lim=(-0.02, 0.82),
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

    for ax in (ax_sim, ax_cer):
        leg = ax.legend(handles=[scicom_dot, other_dot, scicom_line, region_patch],
                        fontsize=8, facecolor='#ffffff', edgecolor='#cccccc',
                        labelcolor=ss['label_color'],
                        loc='upper left', framealpha=0.9)

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
