"""From subtitle files (.srt) to labelled pairs: read, clean, join split sentences, measure the gap, label, balance."""
import re
from pathlib import Path

import pandas as pd

# File name of the .srt (without ending) -> film. 'oppenheimer' was the thirteenth film; it and '6888' are left out of the
# final data set because data problems in their files distort the short-silence class.
FILMS = {
    "6888": "The Six Triple Eight", "basis": "On the Basis of Sex", "benefits": "Friends with Benefits",
    "conviction": "Conviction", "crazystupid": "Crazy, Stupid, Love", "hacksaw": "Hacksaw Ridge", "harriet": "Harriet",
    "help": "The Help", "hidden": "Hidden Figures", "immitation": "The Imitation Game", "king": "King Richard",
    "pslove": "P.S. I Love You",
}
EXCLUDED = {"6888", "oppenheimer"}
THRESHOLD_SEC = 0.7
SEED = 42

TIMESTAMP_RE = re.compile(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})")
TAG_RE = re.compile(r"</?[^>]+>")
LEADING_QUOTES = "\"“”«‹‘’'"


def to_seconds(stamp: str) -> float:
    hours, minutes, rest = stamp.strip().split(":")
    seconds, millis = rest.split(",")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def parse_srt(path: Path) -> list[dict]:
    """One dict per subtitle block: id, start and end time stamp, text without styling tags."""
    rows = []
    for block in re.split(r"\n\s*\n", Path(path).read_text(encoding="utf-8", errors="ignore").strip()):
        lines = block.strip().splitlines()
        if not lines:
            continue
        numbered = bool(re.fullmatch(r"\d+", lines[0].strip()))
        if numbered and len(lines) < 2:
            continue
        match = TIMESTAMP_RE.match(lines[1 if numbered else 0].strip())
        if not match:
            continue
        text = " ".join(line.strip() for line in lines[2 if numbered else 1:] if line.strip())
        rows.append({
            "ID": lines[0].strip() if numbered else str(len(rows) + 1),
            "start": match.group(1), "end": match.group(2),
            "text": TAG_RE.sub("", text).strip(),
        })
    return rows


def continues(current: str, following: str) -> bool:
    """A line goes on in the next block if it has no final . ? ! and the next one starts in lower case or with I."""
    if not current or not following or current.rstrip()[-1:] in {".", "?", "!"}:
        return False
    start = following.lstrip().lstrip(LEADING_QUOTES)
    return bool(start) and ((start[0].isalpha() and start[0].islower()) or start[0] == "I")


def join_split_lines(rows: list[dict]) -> list[dict]:
    """First repair: drop song lines, then join blocks that belong to one utterance."""
    rows = [r for r in rows if "♪" not in r["text"]]
    joined, i = [], 0
    while i < len(rows):
        base, j = rows[i].copy(), i + 1
        while j < len(rows) and continues(base["text"], rows[j]["text"]):
            base["text"] = (base["text"].rstrip() + " " + rows[j]["text"].lstrip()).strip()
            base["end"] = rows[j]["end"]
            j += 1
        joined.append(base)
        i = j
    return joined


def measure_gaps(rows: list[dict], film: str) -> pd.DataFrame:
    """Pairs of neighbouring lines. Gap = start of the second line minus end of the first; overlaps count as 0."""
    lines = sorted(
        ({**r, "start": round(to_seconds(r["start"]), 3), "end": round(to_seconds(r["end"]), 3)} for r in rows),
        key=lambda r: (r["start"], r["ID"]),
    )
    return pd.DataFrame([{
        "file_name": film, "sub1_id": a["ID"], "sub1_text": a["text"], "sub2_text": b["text"],
        "gap_sec": round(max(0.0, b["start"] - a["end"]), 3),
        "sub1_start_sec": a["start"], "sub1_end_sec": a["end"],
        "sub2_start_sec": b["start"], "sub2_end_sec": b["end"],
    } for a, b in zip(lines, lines[1:])])


ELLIPSIS_END_RE = re.compile(r'\.\.{2,}"?\s*$')
ELLIPSIS_STRIP_RE = re.compile(r'\.\.{2,}(?="?\s*$)')


def _goes_on(text: str) -> bool:
    s = text.lstrip()
    return s.startswith("...") or s[:1].islower() or (s[:1] == '"' and s[1:2].islower())


def join_ellipsis_lines(pairs: pd.DataFrame) -> pd.DataFrame:
    """Second repair: '... and then' continues the line before it, so the pause between them is not a turn gap."""
    pairs = pairs.sort_values(["file_name", "sub1_start_sec"]).reset_index(drop=True)
    rows, out, i = pairs.to_dict("records"), [], 0
    while i < len(rows):
        row, nxt = rows[i], rows[i + 1] if i + 1 < len(rows) else None
        if (nxt is not None and row["file_name"] == nxt["file_name"]
                and ELLIPSIS_END_RE.search(row["sub1_text"]) and _goes_on(row["sub2_text"])
                and row["sub2_text"] == nxt["sub1_text"] and abs(row["sub2_start_sec"] - nxt["sub1_start_sec"]) < 1e-3):
            first = ELLIPSIS_STRIP_RE.sub("", row["sub1_text"]).rstrip()
            out.append({**nxt, "sub1_text": (first + " " + row["sub2_text"].lstrip()).strip(), "sub1_id": row["sub1_id"],
                        "sub1_start_sec": row["sub1_start_sec"], "sub1_end_sec": row["sub2_end_sec"]})
            i += 2
        else:
            out.append(row)
            i += 1
    return pd.DataFrame(out)


def build_pairs(srt_dir: Path) -> pd.DataFrame:
    """All labelled pairs of all films, before balancing."""
    frames = [measure_gaps(join_split_lines(parse_srt(path)), path.stem) for path in sorted(Path(srt_dir).glob("*.srt"))]
    pairs = join_ellipsis_lines(pd.concat(frames, ignore_index=True))
    pairs["silence_label"] = (pairs["gap_sec"] >= THRESHOLD_SEC).astype(int)
    return pairs


def balance(pairs: pd.DataFrame) -> pd.DataFrame:
    """Leave out the two problem films and draw the same number of pairs from both classes."""
    pairs = pairs[~pairs["file_name"].isin(EXCLUDED)]
    short, long = pairs[pairs["silence_label"] == 0], pairs[pairs["silence_label"] == 1]
    n = min(len(short), len(long))
    drawn = pd.concat([short.sample(n=n, random_state=SEED), long.sample(n=n, random_state=SEED)])
    return drawn.sample(frac=1, random_state=SEED).reset_index(drop=True)
