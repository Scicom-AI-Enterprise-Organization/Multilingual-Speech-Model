"""
Offline regression tests for postnormalizer.py.

Each case pins one deterministic rule that was derived from the
``text`` -> ``postprocessed_text`` mapping in the dataset
``Scicom-intl/Normalized-Multilingual-TTS``.

Run:  python test_postnormalizer.py       (or: pytest test_postnormalizer.py)
"""

from postnormalizer import PostNormalizer, normalize, detect_script

# (input, expected, description)
CASES = [
    # --- alignment marker stripping (Chinese subsets use %/$ as token markers) ---
    ("各种%境内%外资$看好%一线%城市%楼市$", "各种境内外资看好一线城市楼市。",
     "strip %/$ markers, add ideographic full stop"),
    ("童趣%游戏%面包%超人%第一季$", "童趣游戏面包超人第一季。",
     "strip markers between/after CJK"),
    # --- genuine percent / currency must survive (glued to a digit) ---
    ("the rate is 50% higher", "The rate is 50% higher.", "keep 50%"),
    ("these cost $100 total", "These cost $100 total.", "keep $100"),
    ("price is €50 but ₹ alone", "Price is €50 but alone.",
     "currency symbol kept only when glued to a digit"),

    # --- capitalization ---
    ("menceritakan segala", "Menceritakan segala.", "Latin: cap first + terminal ."),
    ("το σπίτι είναι μεγάλο", "Το σπίτι είναι μεγάλο.", "Greek: cap first"),
    ("днес е хубав ден!!!", "Днес е хубав ден!", "Cyrillic: cap first, collapse !!!"),
    ("first sentence. second one? third!", "First sentence. Second one? Third!",
     "capitalize after sentence-final . ? !"),
    ("at 2:00 p.m. on the dot", "At 2:00 p.m. on the dot.",
     "abbreviation chain p.m. does not end a sentence"),
    ("tak potrzebne m.in. w okresie", "Tak potrzebne m.in. w okresie.",
     "multi-letter abbreviation chain m.in. guarded"),
    ("sent by P.U. tomorrow", "Sent by P.U. tomorrow.", "initials P.U. guarded"),

    # --- no capitalization for caseless scripts ---
    ("私に出来る事なら、サービスして差し上げたいので",
     "私に出来る事なら、サービスして差し上げたいので。", "Japanese: no cap, 。 terminal"),
    ("무슨 기준으로 어디는 그렇게", "무슨 기준으로 어디는 그렇게.",
     "Korean: keep spaces (unlike CJK), add . terminal"),
    ("അനവധി പ്രശസ്ത വ്യക്തിത്വങ്ങൾ എത്തി", "അനവധി പ്രശസ്ത വ്യക്തിത്വങ്ങൾ എത്തി.",
     "Malayalam: add . after combining vowel sign"),

    # --- disallowed symbol / quote / bracket / emoji removal ---
    ("这是…なるほど 「引用」 🌬♥", "这是、なるほど引用。",
     "CJK ellipsis -> 、, drop corner brackets & emoji"),
    ('  spaced   out ,  weird  quote "x" (paren) [b] {c}  ',
     "Spaced out, weird quote x paren b c.", "collapse ws, space-before-punct, drop quotes/brackets"),
    ("tone˥˦ bars˧", "Tone bars.", "drop Chao tone-bar modifier letters"),
    ("hello ​world﻿ test", "Hello world test.", "drop zero-width / BOM"),
    ("می‌خواهم", "میخواهم", "ZWNJ merges away; 1-word fragment unterminated"),
    ("Agwara anyị <?> ka anyị si", "Agwara anyị ka anyị si.",
     "bracket group with no word content removed outright"),

    ("Bitte. . Der Geometer tat", "Bitte. Der Geometer tat.",
     "spaced/double dots collapse to one (reference: 59:1)"),

    # --- ellipsis handling ---
    ("You know... No, never mind.", "You know... No, never mind.",
     "ASCII ... kept verbatim in non-CJK text"),
    ("wait…what", "Wait. What.", "unicode … -> period in non-CJK text"),
    ("「……はい」", "はい。", "leading ellipsis dropped"),

    # --- fullwidth handling ---
    ("senkoku kii ta 、 kotoba wo 、 nochi de 。", "Senkoku kii ta, kotoba wo, nochi de.",
     "romanized Japanese: fullwidth punct folds to ASCII (no CJK chars)"),
    ("「おなにー？　おなにーってなんだ？」", "おなにー？ おなにーってなんだ？",
     "CJK text keeps fullwidth punct AND the space after it"),

    # --- punctuation folding onto the canonical ASCII set ---
    ("وشمل المعرض لوحات فنية، من رسم", "وشمل المعرض لوحات فنية, من رسم.",
     "Arabic comma -> ,  (dataset uses ASCII punctuation)"),

    # --- Arabic: Quranic annotation signs ---
    ("وَيَنْـَٔوْنَ عَنْهُۭ", "وَيَنْـَٔوْنَ عَنْهُ.", "word-attached Quranic sign stripped"),
    ("مِن قَبْلُ ۚ وَهُوَ", "مِن قَبْلُ ۚ وَهُوَ.", "standalone verse-pause mark kept"),
    ("كِـتَـاب", "كِـتَـاب", "tatweel kept; 1-word Arabic fragment unterminated"),
    ("وَيَسْتَغْفِرُونَهُۥ لَهُ", "وَيَسْتَغْفِرُونَهُۥ لَهُ.",
     "small waw U+06E5 kept (pronounced letter, not an annotation)"),

    # --- glued punctuation ---
    ("word,word and more:text", "Word, word and more: text.",
     "insert missing space after clause punctuation"),

    # --- short fragments: no terminal in Devanagari/Cyrillic/Arabic ---
    ("зона", "Зона", "1-word Cyrillic fragment: no terminal"),
    ("пять слов в этом предложении есть", "Пять слов в этом предложении есть.",
     "full Cyrillic sentence still gets terminal"),
    ("पाँच सौ", "पाँच सौ", "2-word Devanagari fragment: no terminal"),
    ("зона.", "Зона.", "already-terminated fragment keeps its mark"),
    ("kata", "Kata.", "1-word Latin still gets terminal (measured: coin flip)"),

    # --- numeric-only strings are not sentences: no terminal, tidy hyphen ---
    ("0896 - 3822075", "0896-3822075", "phone number: no terminal ., collapse hyphen spaces"),
    ("123", "123", "bare number unchanged"),

    # --- misc ---
    ("wait — what? really!!", "Wait — what? Really!", "keep em dash, collapse !!, cap after ?"),
    (None, "", "None -> empty"),
    ("", "", "empty -> empty"),
    ("   ", "", "whitespace only -> empty"),
]


def run():
    failed = 0
    for src, expected, desc in CASES:
        got = normalize(src)
        ok = got == expected
        if not ok:
            failed += 1
            print(f"FAIL [{desc}]")
            print(f"   in : {src!r}")
            print(f"   exp: {expected!r}")
            print(f"   got: {got!r}")
    # config flags
    pn = PostNormalizer(capitalize=False, add_terminal=False)
    assert pn.normalize('hello world "q" 🌬') == "hello world q", "clean-only flags"
    assert PostNormalizer(strip_symbols=False).normalize("a 🌬") != "a", "strip toggle"
    # script detection
    assert detect_script("hello") == "LATIN"
    assert detect_script("你好") == "HAN"
    assert detect_script("안녕") == "HANGUL"
    assert detect_script("مرحبا") == "ARABIC"
    assert detect_script("123 !!!") == "UNKNOWN"

    total = len(CASES)
    print(f"\n{total - failed}/{total} rule cases passed; config/detection assertions passed.")
    if failed:
        raise SystemExit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    run()
