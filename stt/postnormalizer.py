"""
Rule-based multilingual post-normalizer for TTS transcripts.

Derived from the transformation observed in the dataset
``Scicom-intl/Normalized-Multilingual-TTS`` (all ~190 subsets), specifically the
mapping from the raw ``text`` column to the cleaned ``postprocessed_text`` column.

That reference mapping was originally produced by an LLM (Qwen2.5-72B) doing
punctuation + capitalization restoration.  Part of what it does is *deterministic*
(strip alignment markers, drop disallowed symbols, normalize whitespace,
capitalize sentence starts, add a sentence-final mark) and part of it is
*semantic* (deciding where a mid-sentence comma belongs, turning ``.`` into ``;``,
inferring that a sentence is a question).

This module reproduces the **deterministic** part with pure rules.  It never
invents mid-sentence commas or guesses interrogatives, because a rule engine
cannot do that reliably across 100+ languages -- doing so would corrupt text.

Every rule below was validated against the dataset golds; data-derived
behaviours worth knowing about:

* ``%`` / ``$`` alignment markers are stripped only when NOT glued to a digit
  (``50%`` and ``$100`` survive).
* Stripped symbols become word separators, except after CJK punctuation and for
  invisible format characters (ZWNJ inside Persian/Arabic words merges away).
* Korean keeps its inter-word spaces; only Han/Kana runs are space-joined.
* Word-attached Quranic annotation signs (the Mn marks in U+06D6..U+06ED) are
  dropped; standalone verse-pause marks, the pronounced small waw/yeh
  (U+06E5/U+06E6) and the tatweel are kept -- the reference keeps all three.
* ``…`` becomes an ideographic comma between CJK characters and a period
  elsewhere; ASCII ``...`` runs are kept verbatim in non-CJK text (the
  reference keeps them 2:1).
* Fullwidth punctuation folds to ASCII only when the text contains no CJK
  characters at all (e.g. romanized Japanese); otherwise it is kept, including
  a following space (the reference keeps ``？ `` spacing).
* Sentence-internal capitalization is restored after ``. ! ?`` with guards for
  initials (``P.U.``), abbreviation chains (``p.m.``, ``e.g.``, ``m.in.``) and
  ellipses.
* Bracket groups with no word content (``<?>``, ``[?]``) are removed outright.
* Short unpunctuated fragments (single words, list items) stay unterminated in
  Devanagari/Cyrillic/Arabic, matching the reference (93%/77%/66%).

Design goals
------------
* stdlib only (``re`` + ``unicodedata``); no models, no network, fast enough for
  millions of rows.
* multilingual by Unicode script detection -- no per-language config required,
  though a ``script`` hint can be passed to skip detection.
* every transform is a toggleable flag, so callers can use it purely as a
  cleaner (no capitalization / no terminal punctuation) if they prefer.

Usage
-----
    >>> from postnormalizer import normalize
    >>> normalize('各种%境内%外资$看好%一线%城市%楼市$')
    '各种境内外资看好一线城市楼市。'
    >>> normalize('menceritakan segala')
    'Menceritakan segala.'
    >>> normalize('the rate is 50% higher "wow" 🌬')
    'The rate is 50% higher wow.'

CLI
---
    python postnormalizer.py "some raw text"
    echo "line1\\nline2" | python postnormalizer.py      # one line per stdin line
"""

from __future__ import annotations

import re
import sys
import unicodedata

__all__ = ["PostNormalizer", "normalize", "detect_script"]

# ---------------------------------------------------------------------------
# Script detection
# ---------------------------------------------------------------------------

# Unicode block ranges -> coarse script label.  Order matters only for lookup
# speed; ranges are non-overlapping.
_SCRIPT_RANGES = [
    ("HAN",       (0x4E00, 0x9FFF)),
    ("HAN",       (0x3400, 0x4DBF)),   # CJK ext A
    ("HAN",       (0x20000, 0x2A6DF)), # CJK ext B
    ("HAN",       (0xF900, 0xFAFF)),   # compatibility ideographs
    ("HIRAGANA",  (0x3040, 0x309F)),
    ("KATAKANA",  (0x30A0, 0x30FF)),
    ("KATAKANA",  (0x31F0, 0x31FF)),
    ("HANGUL",    (0xAC00, 0xD7A3)),
    ("HANGUL",    (0x1100, 0x11FF)),
    ("HANGUL",    (0x3130, 0x318F)),
    ("ARABIC",    (0x0600, 0x06FF)),
    ("ARABIC",    (0x0750, 0x077F)),
    ("ARABIC",    (0x08A0, 0x08FF)),
    ("ARABIC",    (0xFB50, 0xFDFF)),
    ("ARABIC",    (0xFE70, 0xFEFF)),
    ("HEBREW",    (0x0590, 0x05FF)),
    ("CYRILLIC",  (0x0400, 0x04FF)),
    ("CYRILLIC",  (0x0500, 0x052F)),
    ("GREEK",     (0x0370, 0x03FF)),
    ("GREEK",     (0x1F00, 0x1FFF)),
    ("ARMENIAN",  (0x0530, 0x058F)),
    ("GEORGIAN",  (0x10A0, 0x10FF)),
    ("DEVANAGARI",(0x0900, 0x097F)),
    ("BENGALI",   (0x0980, 0x09FF)),
    ("GURMUKHI",  (0x0A00, 0x0A7F)),
    ("GUJARATI",  (0x0A80, 0x0AFF)),
    ("ORIYA",     (0x0B00, 0x0B7F)),
    ("TAMIL",     (0x0B80, 0x0BFF)),
    ("TELUGU",    (0x0C00, 0x0C7F)),
    ("KANNADA",   (0x0C80, 0x0CFF)),
    ("MALAYALAM", (0x0D00, 0x0D7F)),
    ("SINHALA",   (0x0D80, 0x0DFF)),
    ("THAI",      (0x0E00, 0x0E7F)),
    ("LAO",       (0x0E80, 0x0EFF)),
    ("TIBETAN",   (0x0F00, 0x0FFF)),
    ("MYANMAR",   (0x1000, 0x109F)),
    ("ETHIOPIC",  (0x1200, 0x137F)),
    ("LATIN",     (0x0041, 0x024F)),
    ("LATIN",     (0x1E00, 0x1EFF)),   # Latin extended additional (Vietnamese)
]

# Scripts that have upper/lower case (so capitalization rules apply).
_CASED_SCRIPTS = {"LATIN", "CYRILLIC", "GREEK", "ARMENIAN"}

# Scripts whose sentence-final mark is the ideographic full stop.
_IDEOGRAPHIC_SCRIPTS = {"HAN", "HIRAGANA", "KATAKANA"}


def _char_script(ch: str) -> str | None:
    o = ord(ch)
    for name, (lo, hi) in _SCRIPT_RANGES:
        if lo <= o <= hi:
            return name
    return None


def _is_cjk(ch: str) -> bool:
    """Han or Kana (space-less scripts); Hangul is deliberately NOT included."""
    o = ord(ch)
    return (0x3400 <= o <= 0x4DBF) or (0x4E00 <= o <= 0x9FFF) or (0x3040 <= o <= 0x30FF)


def detect_script(text: str) -> str:
    """Return the dominant alphabetic script of *text* (e.g. ``"LATIN"``, ``"HAN"``).

    Only characters carrying a script (letters) vote; punctuation/digits/spaces
    are ignored.  Returns ``"UNKNOWN"`` when nothing votes.
    """
    counts: dict[str, int] = {}
    for ch in text:
        if ch.isalpha():
            s = _char_script(ch)
            if s:
                counts[s] = counts.get(s, 0) + 1
    if not counts:
        return "UNKNOWN"
    return max(counts, key=counts.get)


# ---------------------------------------------------------------------------
# Character rule tables
# ---------------------------------------------------------------------------

# Punctuation we KEEP (everything else in Unicode category P* is stripped).
# Note: single quotes / apostrophes are kept (contractions: don't, l'homme),
# but double quotes and guillemets are stripped.
_KEEP_PUNCT = set(".,!?;:-'’‘—–·"       # . , ! ? ; : - ' ' ' — – ·
                  "。，、？！；：・")  # 。 ， 、 ？ ！ ； ： ・

# Explicit punctuation remapping applied *before* the keep-filter, so that
# script-specific punctuation is folded onto the canonical allowed set.
_PUNCT_MAP = {
    # Arabic (the reference corpus uses ASCII punctuation for Arabic)
    "،": ",",   # ، arabic comma
    "؛": ";",   # ؛ arabic semicolon
    "؟": "?",   # ؟ arabic question mark
    "٫": ",",   # ٫ arabic decimal separator -> comma (rare)
    "٬": ",",   # ٬ arabic thousands separator
    "۔": ".",   # ۔ urdu full stop
    "؞": ".",   # ؞ arabic triple dot punctuation
    # Indic danda
    "।": ".",   # । devanagari danda
    "॥": ".",   # ॥ double danda
    # Ethiopic
    "።": ".",   # ። ethiopic full stop
    "፡": " ",   # ፡ ethiopic wordspace
    "፣": ",",   # ፣ ethiopic comma
    "፤": ";",   # ፤ ethiopic semicolon
    "፥": ":",   # ፥ ethiopic colon
    "፦": ":",   # ፦ ethiopic preface colon
    "፧": "?",   # ፧ ethiopic question mark
    "፨": ".",   # ፨ ethiopic paragraph separator
    # Armenian
    "։": ".",   # ։ armenian full stop
    "՜": "!",   # ՜ armenian exclamation mark
    "՞": "?",   # ՞ armenian question mark
    # Myanmar
    "။": ".",   # ။ myanmar section
    "၊": ",",   # ၊ myanmar little section
    # dashes / bars -> em dash
    "―": "—",  # ― horizontal bar
    "‒": "—",  # ‒ figure dash
    # ideographic space handled by whitespace pass, but map fullwidth tilde away
    "～": "",        # ～ fullwidth tilde (prolongation noise)
    "〜": "",        # 〜 wave dash
}

# Currency symbols survive only when glued to a digit (mirrors the $ rule).
_CURRENCY = set("€£¥₹₩₫฿₴₪")

# Sentence-final marks already present -> don't append a terminal mark.
_SENTENCE_FINAL = set(".!?。！？…")  # . ! ? 。 ！ ？ …
# Trailing marks that should be dropped before adding a terminal mark
# (a clause comma should not end an utterance).
_TRAILING_STRIP = set(",，、·:：")   # , ， 、 · : ：
# For these scripts the reference leaves short unpunctuated fragments (single
# words, list items) without a terminal mark: Devanagari 93% of 1-word inputs,
# Cyrillic 77%, Arabic 66%.  Value = max word count treated as a fragment.
_FRAGMENT_LIMIT = {"DEVANAGARI": 2, "CYRILLIC": 1, "ARABIC": 1}

# ---------------------------------------------------------------------------
# Precompiled regexes
# ---------------------------------------------------------------------------

# alignment markers: strip % and $ unless glued to a digit (50%, $100)
_MARK_PCT = re.compile(r"(?<!\d)%(?!\d)")
_MARK_DOL = re.compile(r"(?<!\d)\$(?!\d)")

_WS = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?。，、？！；：])")
# spaces around a hyphen between digits (phone numbers, ranges): 0896 - 3822 -> 0896-3822
_HYPHEN_DIGITS = re.compile(r"(\d)\s*-\s*(\d)")
# collapse a run of the same mark: !!! -> !, ？？ -> ？.  Dot runs of 3+ are
# ellipses, not typos -- the reference keeps '...' 2:1 -- so '.' is NOT in this
# class; exactly-two-dot runs ARE artifacts (reference collapses them 59:1).
_REPEAT_PUNCT = re.compile(r"([!?。！？，、,])\1+")
_TWO_DOTS = re.compile(r"(?<!\.)\.\.(?!\.)")
# space sitting between two ideographic characters -> remove.
# Only Han + Kana (Chinese/Japanese) are space-less; Korean/Hangul DOES use
# inter-word spaces, so it is deliberately excluded here.
_CJK = r"㐀-䶿一-鿿぀-ヿ"
_CJK_SPACE = re.compile(rf"(?<=[{_CJK}])\s+(?=[{_CJK}])")

# word-attached Quranic annotation signs (category Mn only); standalone
# verse-pause marks are spared by the lookbehind, and the small waw/yeh
# U+06E5/U+06E6 are excluded -- they are modifier LETTERS (pronounced vowel
# prolongation) and the reference corpus keeps them without exception.
_QURANIC = re.compile(r"(?<=\S)[\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]")
# bracket groups with no word content: annotation markers like <?>, [?], (...)
_EMPTY_BRACKETS = re.compile(r"[<\[({「『【（][\W_]*?[>\])}」』】）]")
# ellipses: unicode chars vs ASCII dot runs are handled differently
_ELL_UNI = re.compile(r"…+|‥+")
_ELL_DOTS = re.compile(r"\.{3,}")
# artifact gaps (sentinels left by stripped symbols) merge after CJK punctuation
_SENTINEL_FW = re.compile("([。，、！？；：·])\x00+")
# fullwidth -> ASCII fold for text with no CJK characters (romanized Japanese &c.)
_FW_TABLE = str.maketrans("。，、！？；：．｡､", ".,,!?;:..,")
# missing space after clause punctuation between letters: "word,word"
_GLUED = re.compile(r"(?<=[^\W\d_])([,;:!?.])(?=[^\W\d_])", re.UNICODE)
# capitalization after sentence-final punctuation
_CAP_AFTER = re.compile(r"([.!?。！？])(\s+)(\S)")


class PostNormalizer:
    """Configurable rule-based post-normalizer.

    Parameters
    ----------
    remove_markers:
        Strip ``%`` / ``$`` alignment markers (keeping genuine ``50%`` / ``$100``).
    strip_symbols:
        Remove emojis, tone bars, quotes, brackets and other disallowed symbols,
        fold script-specific punctuation onto the canonical set, drop
        word-attached Quranic signs, resolve ellipses, and fold fullwidth
        punctuation to ASCII in fully non-CJK text.
    normalize_whitespace:
        Collapse whitespace, drop space-before-punctuation, join CJK runs, and
        insert the missing space after glued clause punctuation ("word,word").
    capitalize:
        Capitalize the first letter and letters after sentence-final ``. ! ?``
        for cased scripts (Latin/Cyrillic/Greek/Armenian), with guards for
        initials and abbreviation chains (``P.U.``, ``p.m.``, ``e.g.``, ``m.in.``).
    add_terminal:
        Append a sentence-final mark (``。`` for CJK, ``.`` otherwise) when the
        utterance does not already end in one.  Skipped for strings with no
        detectable script (phone numbers, IDs) and for short unpunctuated
        fragments in Devanagari/Cyrillic/Arabic.
    """

    def __init__(
        self,
        remove_markers: bool = True,
        strip_symbols: bool = True,
        normalize_whitespace: bool = True,
        capitalize: bool = True,
        add_terminal: bool = True,
    ):
        self.remove_markers = remove_markers
        self.strip_symbols = strip_symbols
        self.normalize_whitespace = normalize_whitespace
        self.capitalize = capitalize
        self.add_terminal = add_terminal

    # -- individual passes -------------------------------------------------

    @staticmethod
    def _resolve_ellipsis(m: re.Match, uni: bool) -> str:
        """Context-sensitive replacement for one ellipsis run (see docstring)."""
        s, i, j = m.string, m.start(), m.end()
        k = j
        while k < len(s) and s[k].isspace():
            k += 1
        p = i - 1
        while p >= 0 and s[p].isspace():
            p -= 1
        if p < 0 or not s[p].isalnum():
            return ""                       # leading (or after quote/bracket)
        if k < len(s) and not (s[k].isalnum() or s[k] in "%$'"):
            return ""                       # followed by punctuation
        if _is_cjk(s[p]):
            if k >= len(s):
                return ""                   # trailing -> terminal pass adds 。
            if _is_cjk(s[k]):
                return "、"                 # CJK pause -> ideographic comma
        # non-CJK: '…' collapses to a period, ASCII '...' stays verbatim
        return "." if uni else m.group(0)

    @classmethod
    def _ellipses(cls, text: str) -> str:
        """Resolve ``…`` / ``...`` by context (see module docstring)."""
        text = _ELL_UNI.sub(lambda m: cls._resolve_ellipsis(m, True), text)
        return _ELL_DOTS.sub(lambda m: cls._resolve_ellipsis(m, False), text)

    def _strip_symbols(self, text: str) -> str:
        """Map known punctuation, then drop disallowed symbols/punctuation.

        Dropped *visible* characters leave a ``\\x00`` sentinel so that words
        they separated do not merge; invisible format characters (ZWNJ, ZWJ,
        BOM) vanish outright so words they sat inside stay whole.  Sentinels
        are resolved into spaces (or nothing, after CJK punctuation) later.
        """
        out = []
        n = len(text)
        for i, ch in enumerate(text):
            if ch in _PUNCT_MAP:
                out.append(_PUNCT_MAP[ch])
                continue
            if ch in "%$":                      # legit percent/currency: markers
                out.append(ch)                  # already removed in the marker pass
                continue
            if ch in _CURRENCY:
                if (i > 0 and text[i-1].isdigit()) or (i + 1 < n and text[i+1].isdigit()):
                    out.append(ch)
                    continue
            cat = unicodedata.category(ch)
            c0 = cat[0]
            if c0 in ("L", "M", "N"):          # letters, combining marks, numbers
                out.append(ch)
            elif ch in (" ", "\t", "\n", "\r") or cat == "Zs":
                out.append(" ")                 # normalize any space separator
            elif c0 == "P":                     # punctuation: keep only allowed
                out.append(ch if ch in _KEEP_PUNCT else "\x00")
            elif cat != "Cf":                   # visible symbol -> word gap
                out.append("\x00")
            # Cf (ZWNJ/ZWJ/BOM/...) -> drop entirely
        # resolve sentinels: merge after CJK punctuation, space elsewhere
        return _SENTINEL_FW.sub(r"\1", "".join(out)).replace("\x00", " ")

    def _whitespace(self, text: str) -> str:
        text = _CJK_SPACE.sub("", text)
        text = _HYPHEN_DIGITS.sub(r"\1-\2", text)
        text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
        text = _WS.sub(" ", text)
        return text.strip()

    @staticmethod
    def _unglue(text: str) -> str:
        """Insert the missing space after clause punctuation between letters."""
        def _g(m: re.Match) -> str:
            p = m.group(1)
            s, i = m.string, m.start(1)
            prev, nxt = s[i-1], s[i+1]
            if _is_cjk(prev) or _is_cjk(nxt):
                return p                        # CJK keeps glued ASCII punct
            if p == ".":
                if prev.isupper() or nxt.isupper():
                    return p                    # initials: U.S., P.U.
                if i >= 2 and not s[i-2].isalpha():
                    return p                    # single-letter token: p.m.
                if i + 2 < len(s) and s[i+2] == ".":
                    return p                    # next is single letter + dot
            return p + " "
        return _GLUED.sub(_g, text)

    def _terminal(self, text: str, script: str, had_final: bool) -> str:
        # only sentences get a terminal mark: skip numeric-only / symbol-only
        # strings (phone numbers, IDs, codes) which carry no detectable script.
        if script in (None, "UNKNOWN"):
            return text
        # drop a trailing clause-mark that should not end an utterance
        while text and text[-1] in _TRAILING_STRIP:
            text = text[:-1].rstrip()
        if not text:
            return text
        if text[-1] in _SENTENCE_FINAL or text[-1] in ";；":
            return text
        # only append after a letter / number / combining-mark (covers Indic vowel
        # signs, which are category Mn/Mc and would fail str.isalnum()); never
        # after a dash, colon or other trailing punctuation/symbol.
        if unicodedata.category(text[-1])[0] not in ("L", "M", "N"):
            return text
        # short unpunctuated fragments stay unterminated in some scripts
        lim = _FRAGMENT_LIMIT.get(script)
        if lim and not had_final and len(text.split()) <= lim:
            return text
        mark = "。" if script in _IDEOGRAPHIC_SCRIPTS else "."
        return text + mark

    def _capitalize(self, text: str) -> str:
        # first letter of the utterance
        for i, ch in enumerate(text):
            if ch.isalpha():
                s = _char_script(ch)
                if s in _CASED_SCRIPTS and ch.islower():
                    text = text[:i] + ch.upper() + text[i + 1:]
                break  # first letter already cased/uppercase or uncased script
        # letters after sentence-final punctuation
        def _c(m: re.Match) -> str:
            p, sp, ch = m.groups()
            s, i = m.string, m.start()
            if ch.islower() and _char_script(ch) in _CASED_SCRIPTS:
                if p == "." and i > 0:
                    if s[i-1] == ".":
                        return m.group(0)       # after '...' run: ambiguous
                    if s[i-1].isupper() and (i < 2 or not s[i-2].isalpha()):
                        return m.group(0)       # initial like "P.U. x"
                    q = i - 1                   # letter-run before the dot
                    while q >= 0 and s[q].isalpha():
                        q -= 1
                    if q >= 0 and s[q] == ".":
                        return m.group(0)       # abbrev chain: m.in., p.m., e.g.
                return p + sp + ch.upper()
            return m.group(0)
        return _CAP_AFTER.sub(_c, text)

    # -- public API --------------------------------------------------------

    def normalize(self, text: str, script: str | None = None) -> str:
        if text is None:
            return ""
        text = unicodedata.normalize("NFC", str(text))
        # NUL is reserved as the internal word-gap sentinel and is never
        # legitimate text: remove it unconditionally, whatever the config.
        text = text.replace("\x00", "")

        # remember whether the source already ended in a sentence-final mark
        # (used by the fragment rule in _terminal)
        stripped = text.strip()
        had_final = bool(stripped) and stripped[-1] in _SENTENCE_FINAL

        if self.remove_markers:
            text = _MARK_PCT.sub("", text)
            text = _MARK_DOL.sub("", text)

        if script is None:
            script = detect_script(text)

        if self.strip_symbols:
            text = _QURANIC.sub("", text)
            text = _EMPTY_BRACKETS.sub(" ", text)
            text = self._ellipses(text)
            if script not in _IDEOGRAPHIC_SCRIPTS and not any(map(_is_cjk, text)):
                text = text.translate(_FW_TABLE)
            text = self._strip_symbols(text)
        if self.normalize_whitespace:
            text = self._whitespace(text)

        # collapse repeats BEFORE unglue so pairs made adjacent by the collapse
        # (e.g. "A!!c" -> "A!c") still receive their separating space
        text = _REPEAT_PUNCT.sub(r"\1", text)
        text = _TWO_DOTS.sub(".", text)
        if self.strip_symbols:
            # punctuation folding (e.g. doubled ۔/। -> ..) can synthesize new
            # dot runs after the first ellipsis pass; resolve them identically.
            # A no-op for runs the first pass already chose to keep.
            text = _ELL_DOTS.sub(lambda m: self._resolve_ellipsis(m, False), text)

        if self.normalize_whitespace:
            text = self._unglue(text)
        text = text.strip()

        if not text:
            return text

        if self.capitalize:
            text = self._capitalize(text)
        if self.add_terminal:
            text = self._terminal(text, script, had_final)
        return text

    __call__ = normalize


# module-level default instance / convenience function
_DEFAULT = PostNormalizer()


def normalize(text: str, script: str | None = None) -> str:
    """Normalize *text* with default settings.  See :class:`PostNormalizer`."""
    return _DEFAULT.normalize(text, script=script)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(normalize(" ".join(sys.argv[1:])))
    else:
        for line in sys.stdin:
            print(normalize(line.rstrip("\n")))
