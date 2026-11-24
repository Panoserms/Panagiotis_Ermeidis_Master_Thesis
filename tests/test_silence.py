from pathlib import Path

import pandas as pd
import pytest

from silence import subtitles
from silence.feature_groups import BOOLEAN_FEATURES, GROUPS

DATA = Path(__file__).parent.parent / "data"


def block(start, end, text):
    return {"ID": "1", "start": start, "end": end, "text": text}


def test_a_sentence_over_two_blocks_is_joined():
    rows = [block("00:00:01,000", "00:00:02,000", "I want my friend back,"),
            block("00:00:02,284", "00:00:04,000", "because I miss her."),
            block("00:00:05,000", "00:00:06,000", "Really?")]
    joined = subtitles.join_split_lines(rows)
    assert [r["text"] for r in joined] == ["I want my friend back, because I miss her.", "Really?"]
    gaps = subtitles.measure_gaps(joined, "film")["gap_sec"].tolist()
    assert gaps == [1.0]


def test_song_lines_are_dropped():
    assert subtitles.join_split_lines([block("00:00:01,000", "00:00:02,000", "♪ la la ♪")]) == []


def test_overlap_counts_as_zero():
    rows = [block("00:00:01,000", "00:00:03,000", "One."), block("00:00:02,500", "00:00:04,000", "Two.")]
    assert subtitles.measure_gaps(rows, "film")["gap_sec"].tolist() == [0.0]


def test_ellipsis_continuation_is_joined():
    rows = [block("00:00:01,000", "00:00:02,000", "I was going to..."), block("00:00:02,100", "00:00:03,000", "...tell you."),
            block("00:00:04,000", "00:00:05,000", "Tell me what?")]
    pairs = subtitles.join_ellipsis_lines(subtitles.measure_gaps(rows, "film"))
    assert pairs["sub1_text"].tolist() == ["I was going to ...tell you."]
    assert pairs["gap_sec"].tolist() == [1.0]


def test_feature_groups():
    assert {name: len(group) for name, group in GROUPS.items()} == {
        "Syntactic": 172, "Lexical-semantic": 27, "Structural": 20, "Affect": 68}
    everything = sum(GROUPS.values(), [])
    assert len(set(everything)) == 287 and set(BOOLEAN_FEATURES) <= set(everything) and len(BOOLEAN_FEATURES) == 32


@pytest.mark.parametrize("part, n", [("train", 10376), ("validation", 1297), ("test", 1297)])
def test_tables_hold_numbers_only(part, n):
    table = pd.read_csv(DATA / f"features_{part}.csv.gz", dtype={"film": str})
    assert table.shape == (n, 290)
    assert table.drop(columns="film").apply(pd.api.types.is_numeric_dtype).all()
    assert ((table["gap_sec"] >= 0.7).astype(int) == table["silence_label"]).all()


def test_handcrafted_features():
    features = pytest.importorskip("silence.features")
    assert features.count_syllables("the") == 1 and features.count_syllables("could") == 1 and features.count_syllables("moment") == 2
    assert features.non_verbal_score("[SIGHS] Fine.") == 1
    assert features.sluice_features("Why?")["is_bare_wh"] == 1
    assert features.sluice_features("I would, but...") == {"is_bare_wh": 0, "ends_on_cconj": 1, "aposiopesis_score": 1}
    assert features.ellipsis_score("I... I don't know.") >= 1
    assert features.syllables_per_second("[LAUGHS] Come on.", 1.0) == 2
