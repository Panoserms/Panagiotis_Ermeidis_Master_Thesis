"""Linguistic features of one subtitle line (the line before the pause).

Not in this module: the discourse-connective, hedge and intensifier counts and the affect features (VADER, GoEmotions, NRC).
Their values are in the feature tables in data/.
"""
import re

import lftk
import nltk
import numpy as np
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

for _package in ("stopwords", "punkt", "punkt_tab"):  # word lists of NLTK, needed once
    nltk.download(_package, quiet=True)

nlp = spacy.load("en_core_web_sm")
STOP_WORDS = set(stopwords.words("english"))


def _lftk(extractor, code: str) -> float:
    """One LFTK value; 0 where LFTK cannot compute it (for example on an empty line)."""
    try:
        return next(iter(extractor.extract(features=[code]).values()), 0.0)
    except Exception:
        return 0.0


def _content_words(text: str) -> list[str]:
    return [word for word in word_tokenize(text.lower()) if word.isalpha() and word not in STOP_WORDS]


# ── Structural ───────────────────────────────────────────────

BASIC_STATISTICS = {
    "total_number_of_words": "t_word", "total_number_of_stop_words": "t_stopword",
    "total_number_of_punctuations": "n_punct", "total_number_of_syllables": "t_syll",
    "total_number_of_words_more_than_two_syllables": "t_syll2",
    "total_number_of_words_more_than_three_syllables": "t_syll3", "total_number_of_unique_words": "t_uword",
}


def structural_features(text: str) -> dict:
    line, length = text.strip(), len(_content_words(text.strip()))
    extractor = lftk.Extractor(nlp(line))
    values = {name: _lftk(extractor, code) for name, code in BASIC_STATISTICS.items()}
    for name, mark in (("exclamations", "!"), ("question_marks", "?")):
        values[f"num_{name}"] = line.count(mark)
        values[f"ratio_{name}"] = line.count(mark) / length if length else 0.0
    values["sentence_length"] = length
    return values


# ── Lexical-semantic ─────────────────────────────────────────

WORD_FREQUENCY_AND_NAMES = {
    "total_subtlex_us_zipf_of_words": "t_subtlex_us_zipf",
    "average_subtlex_us_zipf_of_words_per_word": "a_subtlex_us_zipf_pw",
    "total_number_of_named_entities": "t_n_ent", "average_number_of_named_entities_per_word": "a_n_ent_pw",
    "total_number_of_named_entities_person": "t_n_ent_person",
    "average_number_of_named_entities_person_per_word": "a_n_ent_person_pw",
    "simple_type_token_ratio": "simp_ttr", "simple_type_token_ratio_no_lemma": "simp_ttr_no_lem",
}

VARIATION_TAGS = [
    ("adjectives","adj"), ("adpositions","adp"), ("adverbs","adv"),
    ("auxiliary","aux"), ("coordinating_conjunctions","cconj"),
    ("determiners","det"), ("interjections","intj"), ("nouns","noun"),
    ("numerals","num"), ("particles","part"), ("pronouns","pron"),
    ("proper_nouns","propn"), ("punctuations","punct"),
    ("subordinating_conjunctions","sconj"), ("verbs","verb"),
]


def lexical_features(text: str) -> dict:
    extractor = lftk.Extractor(nlp(text.strip()))
    values = {name: _lftk(extractor, code) for name, code in WORD_FREQUENCY_AND_NAMES.items()}
    values.update({f"simple_{name}_variation": _lftk(extractor, f"simp_{tag}_var") for name, tag in VARIATION_TAGS})
    return values


# ── Politeness (part of the affect group) ────────────────────

POLITENESS_SINGLE = {
    "please","thanks","thank","sorry","pardon","excuse",
    "sir","ma'am","madam","miss","mister","mr","mr.",
    "mrs","mrs.","ms","ms.","doctor","dr","dr.","doc",
}

POLITENESS_PHRASES = {
    "pretty please","thank you","thanks a lot","many thanks",
    "much appreciated","i'm sorry","i am sorry","so sorry",
    "pardon me","excuse me","may i","could i","could you",
    "would you","would you mind","if you don't mind",
    "if you wouldnt mind","would you please","could you please",
}


def politeness_features(text: str) -> dict:
    words = _content_words(text.strip())
    single = sum(word in POLITENESS_SINGLE for word in words)
    phrases = sum(phrase in text.strip().lower() for phrase in POLITENESS_PHRASES)
    return {
        "num_politeness": single + phrases, "num_politeness_single": single, "num_politeness_phrases": phrases,
        "ratio_politeness": (single + phrases) / len(words) if words else 0.0,
    }


# ── Syntactic: part-of-speech and dependency counts ──────────

POS_TAGS = [
    ("adjectives","adj"), ("adpositions","adp"), ("adverbs","adv"),
    ("auxiliaries","aux"), ("coordinating_conjunctions","cconj"),
    ("determiners","det"), ("interjections","intj"), ("nouns","noun"),
    ("numerals","num"), ("particles","part"), ("pronouns","pron"),
    ("proper_nouns","propn"), ("punctuations","punct"),
    ("subordinating_conjunctions","sconj"), ("verbs","verb"),
]

DEP_LABELS = [
    "ROOT","nsubj","ccomp","punct","dobj","prt","acl","acomp","advcl","advmod",
    "agent","amod","appos","attr","aux","auxpass","case","cc","conj",
    "csubj","csubjpass","dative","dep","det","expl","intj","mark","meta",
    "nmod","npadvmod","nsubjpass","nummod","oprd","parataxis","pcomp","pobj",
    "poss","preconj","predet","prep","quantmod","relcl","xcomp",
]


def count_features(text: str) -> dict:
    doc = nlp(text.strip())
    extractor = lftk.Extractor(doc)
    values = {}
    for name, tag in POS_TAGS:
        values[f"total_number_of_{name}"] = _lftk(extractor, f"n_{tag}")
        values[f"total_number_of_unique_{name}"] = _lftk(extractor, f"n_u{tag}")
        values[f"average_number_of_{name}_per_word"] = _lftk(extractor, f"a_{tag}_pw")
    for label in DEP_LABELS:
        values[label] = sum(token.dep_ == label for token in doc)
    return values


# ── Syntactic: trees, distances, clauses, how the line ends ──

def get_tree_depth(token):
    children = list(token.children)
    if not children:
        return 0
    return 1 + max(get_tree_depth(c) for c in children)


def _safe_mean(lst):
    return float(np.mean(lst)) if lst else 0.0


def _safe_max(lst):
    return max(lst) if lst else 0


def _safe_std(lst):
    return float(np.std(lst)) if len(lst) > 1 else 0.0

def syntactic_complexity_features(doc):
    roots = [t for t in doc if t.dep_ == "ROOT"]
    depths = [get_tree_depth(r) for r in roots]
    branching = [len(list(t.children)) for t in doc if list(t.children)]
    left = sum(1 for t in doc if t.i < t.head.i and t.dep_ != "ROOT")
    right = sum(1 for t in doc if t.i > t.head.i and t.dep_ != "ROOT")
    return {
        "tree_depth_max": _safe_max(depths),
        "tree_depth_mean": _safe_mean(depths),
        "branching_factor_mean": _safe_mean(branching),
        "branching_factor_max": _safe_max(branching),
        "n_embedded_clauses": sum(
            1 for t in doc if t.dep_ in {"ccomp", "xcomp", "advcl", "relcl", "acl"}
        ),
        "left_branching_count": left,
        "right_branching_count": right,
        "left_branching_ratio": left / max(left + right, 1),
    }


def dependency_distance_features(doc):
    distances = [abs(t.i - t.head.i) for t in doc if t.dep_ != "ROOT"]
    feats = {
        "dep_distance_mean": _safe_mean(distances),
        "dep_distance_max": _safe_max(distances),
        "dep_distance_sum": sum(distances),
        "dep_distance_std": _safe_std(distances),
    }
    for dep in ["amod", "nsubj", "dobj", "compound", "advcl", "prep", "nsubjpass", "relcl"]:
        d = [abs(t.i - t.head.i) for t in doc if t.dep_ == dep]
        feats[f"dep_distance_{dep}_mean"] = _safe_mean(d)
        feats[f"dep_distance_{dep}_max"] = _safe_max(d)
    return feats


def passive_voice_features(doc):
    n_pass = sum(
        1 for t in doc if t.dep_ in ("nsubjpass", "nsubj:pass", "auxpass", "aux:pass")
    )
    passive_count = 0
    for token in doc:
        if token.tag_ == "VBN":
            for child in token.children:
                if child.dep_ in ("auxpass", "aux:pass", "aux") and child.lemma_ in ("be", "get"):
                    passive_count += 1
                    break
    n_sents = len(list(doc.sents))
    return {
        "has_passive": int(n_pass > 0),
        "passive_clause_count": passive_count,
        "passive_ratio": passive_count / max(n_sents, 1),
    }


def relative_clause_and_subject_features(doc):
    relcl_in_subject = sum(
        1 for t in doc
        if t.dep_ == "relcl" and t.head.dep_ in ("nsubj", "nsubjpass", "nsubj:pass")
    )
    feats = {
        "relcl_in_subject_np": relcl_in_subject,
        "has_relcl_in_subject": int(relcl_in_subject > 0),
        "subject_np_length": 0,
        "subject_np_depth": 0,
        "subject_has_pp": 0,
        "subject_has_relcl": 0,
    }
    for sent in doc.sents:
        subjects = [
            c for c in sent.root.children
            if c.dep_ in ("nsubj", "nsubjpass", "nsubj:pass")
        ]
        if subjects:
            subj = subjects[0]
            sub_toks = list(subj.subtree)
            feats["subject_np_length"] = len(sub_toks)
            feats["subject_np_depth"] = get_tree_depth(subj)
            feats["subject_has_pp"] = int(any(t.dep_ == "prep" for t in sub_toks))
            feats["subject_has_relcl"] = int(any(t.dep_ == "relcl" for t in sub_toks))
            break
    return feats


def clause_type_features(doc):
    n_main = sum(1 for t in doc if t.dep_ in ("ROOT", "parataxis"))
    n_coordinate = sum(
        1 for t in doc
        if t.dep_ == "conj" and t.head.pos_ == "VERB" and t.pos_ == "VERB"
    )
    n_subordinate = sum(
        1 for t in doc if t.dep_ in {"advcl", "ccomp", "xcomp", "relcl", "acl"}
    )
    total = max(n_main + n_coordinate + n_subordinate, 1)

    last = doc[-1] if len(doc) > 0 else None
    if last and last.pos_ == "PUNCT":
        last = doc[-2] if len(doc) > 1 else None
    ends_at_boundary = 0
    if last:
        ends_at_boundary = int(
            last.dep_ in ("ROOT", "conj", "advcl", "ccomp", "xcomp", "relcl", "parataxis")
            or last.head.dep_ == "ROOT"
        )
    return {
        "n_main_clauses": n_main,
        "n_coordinate_clauses": n_coordinate,
        "n_subordinate_clauses": n_subordinate,
        "total_clauses": n_main + n_coordinate + n_subordinate,
        "main_clause_ratio": n_main / total,
        "coordinate_clause_ratio": n_coordinate / total,
        "subordinate_clause_ratio": n_subordinate / total,
        "ends_at_clause_boundary": ends_at_boundary,
    }


def phrasal_features(doc):
    np_len = [len(chunk) for chunk in doc.noun_chunks]
    vp_sz = [len(list(t.subtree)) for t in doc if t.pos_ == "VERB"]
    return {
        "n_noun_phrases": len(np_len),
        "np_length_mean": _safe_mean(np_len),
        "np_length_max": _safe_max(np_len),
        "n_verb_phrases": len(vp_sz),
        "vp_size_mean": _safe_mean(vp_sz),
        "vp_size_max": _safe_max(vp_sz),
        "n_nested_pps": sum(
            1 for t in doc if t.dep_ == "prep" and t.head.dep_ == "pobj"
        ),
    }


def syntactic_completion_features(doc):
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    if not tokens:
        return {
            "ends_syntactically_complete": 0,
            "ends_with_open_dep": 0,
            "final_token_pos": "NONE",
            "final_token_dep": "NONE",
            "final_bigram_pos": "NONE_NONE",
        }
    last = tokens[-1]
    completing = {"ROOT", "dobj", "attr", "acomp", "pobj", "oprd", "npadvmod", "intj", "xcomp", "ccomp"}
    open_deps = {"prep", "cc", "mark", "aux", "det", "amod", "advmod"}
    bigram = f"{tokens[-2].pos_}_{last.pos_}" if len(tokens) >= 2 else f"START_{last.pos_}"
    return {
        "ends_syntactically_complete": int(last.dep_ in completing or last.head.dep_ == "ROOT"),
        "ends_with_open_dep": int(last.dep_ in open_deps),
        "final_token_pos": last.pos_,
        "final_token_dep": last.dep_,
        "final_bigram_pos": bigram,
    }


def non_canonical_features(doc):
    if not list(doc):
        return {
            "has_fronted_adverbial": 0,
            "has_existential": 0,
            "has_cleft": 0,
            "has_topicalization": 0,
            "n_non_canonical": 0,
        }
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    fronted = int(
        root is not None
        and any(c.dep_ == "advcl" and c.i < root.i for c in root.children)
    )
    existential = int(
        any(t.dep_ == "expl" and t.text.lower() == "there" for t in doc)
    )
    cleft = 0
    for t in doc:
        if t.text.lower() == "it" and t.dep_ == "nsubj":
            if any(c.dep_ == "relcl" for c in t.head.children):
                cleft = 1
                break
    topicalization = 0
    if root:
        subj_i = next(
            (c.i for c in root.children if c.dep_ in ("nsubj", "nsubjpass")), None
        )
        obj_i = next(
            (c.i for c in root.children if c.dep_ in ("dobj", "attr")), None
        )
        if subj_i is not None and obj_i is not None and obj_i < subj_i:
            topicalization = 1
    return {
        "has_fronted_adverbial": fronted,
        "has_existential": existential,
        "has_cleft": cleft,
        "has_topicalization": topicalization,
        "n_non_canonical": fronted + existential + cleft + topicalization,
    }


def grouped_final_position_features(final_token_pos, final_token_dep, final_bigram_pos):
    nominals = {"NOUN", "PROPN", "NUM"}
    verbals = {"VERB", "AUX"}
    modifiers = {"ADJ", "ADV"}
    function = {"ADP", "DET", "CCONJ", "SCONJ", "PART"}
    content = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}
    completing = {"dobj", "pobj", "attr", "acomp", "oprd", "npadvmod", "intj", "xcomp", "ccomp"}
    clausal = {"advcl", "relcl", "acl", "ccomp", "parataxis", "conj"}
    open_d = {"prep", "cc", "mark", "det", "amod", "advmod", "neg", "prt", "case"}
    subject = {"nsubj", "nsubjpass", "expl", "csubj"}
    modifier_d = {"amod", "advmod", "nummod", "npadvmod", "appos"}

    parts = final_bigram_pos.split("_") if "_" in final_bigram_pos else ["NONE", "NONE"]
    bp_first = parts[0]
    bp_last = parts[-1]

    return {
        "final_pos_content_word": int(final_token_pos in content),
        "final_pos_nominal": int(final_token_pos in nominals),
        "final_pos_verbal": int(final_token_pos in verbals),
        "final_pos_function_word": int(final_token_pos in function),
        "final_pos_connector": int(final_token_pos in {"CCONJ", "SCONJ", "ADP"}),
        "final_pos_discourse": int(final_token_pos in {"INTJ", "PUNCT", "X"}),
        "final_dep_completing": int(final_token_dep in completing),
        "final_dep_clausal": int(final_token_dep in clausal),
        "final_dep_open": int(final_token_dep in open_d),
        "final_dep_subject": int(final_token_dep in subject),
        "final_dep_modifier": int(final_token_dep in modifier_d),
        "bigram_verb_to_nominal": int(bp_first in verbals and bp_last in nominals),
        "bigram_ends_nominal": int(bp_last in nominals),
        "bigram_ends_verbal": int(bp_last in verbals),
        "bigram_ends_function": int(bp_last in function),
        "bigram_nominal_to_nominal": int(bp_first in nominals and bp_last in nominals),
        "bigram_modifier_to_modifier": int(bp_first in modifiers and bp_last in modifiers),
        "bigram_single_word": int(bp_first == "START"),
        "bigram_ends_intj": int(bp_last == "INTJ"),
    }


def tree_features(text: str) -> dict:
    doc = nlp(text.strip())
    values = {}
    for part in (syntactic_complexity_features, dependency_distance_features, passive_voice_features,
                 relative_clause_and_subject_features, clause_type_features, phrasal_features):
        values.update(part(doc))
    ending = syntactic_completion_features(doc)
    values["ends_syntactically_complete"] = ending["ends_syntactically_complete"]
    values["ends_with_open_dep"] = ending["ends_with_open_dep"]
    values.update(non_canonical_features(doc))
    values.update(grouped_final_position_features(
        ending["final_token_pos"], ending["final_token_dep"], ending["final_bigram_pos"]))
    return values


# ── Handcrafted for subtitles: ellipsis, sound descriptions, speech rate ──

INTERJECTION_LINES = {
    "hey", "hi", "hello", "yo", "wow", "ow", "ugh", "oops", "huh", "hmm",
    "yay", "boo", "okay", "ok", "bye", "off course", "whoo", "uh", "um",
}

INTERJECTION_NORM_RE = re.compile(
    r'^[\s"\'""'']*(.*?)[\s"\'""'']*[.!?…]*\s*$'
)

ELLIPSIS_FINAL_RE = re.compile(r'\.\.{2,}"?\s*$')
ELLIPSIS_INTERNAL_RE = re.compile(r'\.\.{2,}')
TRUNC_DASH_FINAL_RE = re.compile(r'[–—]\s*$')
TRUNC_DASH_INTERNAL_RE = re.compile(r'[–—]\s+')
ELLIPSIS_RESTART_RE = re.compile(
    r'\b([Ii]|[Ww]e|[Yy]ou|[Hh]e|[Ss]he)\s*\.{2,}\s*\1\b'
)


def is_pure_interjection(text: str) -> bool:
    if not text:
        return False
    m = INTERJECTION_NORM_RE.match(text)
    if not m:
        return False
    core = m.group(1).strip().lower()
    return core in INTERJECTION_LINES


def ellipsis_score(text: str) -> int:
    if text is None:
        text = ""
    s = text.strip()

    if is_pure_interjection(s):
        return 0

    score = 0

    has_final_ellipsis = bool(ELLIPSIS_FINAL_RE.search(s))
    has_any_ellipsis = bool(ELLIPSIS_INTERNAL_RE.search(s))
    if has_any_ellipsis and not has_final_ellipsis:
        score += 1

    if TRUNC_DASH_FINAL_RE.search(s):
        score += 1

    if TRUNC_DASH_INTERNAL_RE.search(s):
        score += 1

    if ELLIPSIS_RESTART_RE.search(s):
        score += 1

    return score


WH_WORDS = {"who", "what", "where", "when", "why", "how", "which", "whom", "whose"}


def sluice_features(text: str) -> dict:
    """Bare wh-words ("Why?") and lines that trail off on a conjunction ("but ...")."""
    tokens = [t for t in nlp(text.strip()) if not t.is_space and not t.is_punct]
    if not tokens:
        return {"is_bare_wh": 0, "ends_on_cconj": 0, "aposiopesis_score": 0}
    last = tokens[-1].pos_
    return {
        "is_bare_wh": int(len(tokens) == 1 and tokens[0].text.lower() in WH_WORDS),
        "ends_on_cconj": int(last == "CCONJ"),
        "aposiopesis_score": int(last in {"CCONJ", "SCONJ"}),
    }


def non_verbal_score(text: str) -> float:
    """[SIGHS], (applause), quoted speech inside a line, and capitals (1 point if all words, 0.5 if some)."""
    if not text.strip():
        return 0
    score = int(bool(re.search(r"\[.+?\]", text))) + int(bool(re.search(r"\(.+?\)", text)))
    if re.search(r'"[^"]*"', text) and re.search(r"[a-zA-Z]", re.sub(r'"[^"]*"', "", text)):
        score += 1
    words = re.findall(r"[a-zA-Z]{2,}", re.sub(r"\(.*?\)", "", re.sub(r"\[.*?\]", "", text)))
    capitals = sum(word.isupper() for word in words)
    if capitals:
        score += 1 if capitals == len(words) else 0.5
    return score


def count_syllables(word: str) -> int:
    """Vowel groups, minus a silent final e, at least 1."""
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    count = len(re.findall(r"[aeiouy]+", word))
    return max(count - int(word.endswith("e") and count > 1), 1)


def syllables_per_second(text: str, duration: float) -> float:
    """Syllables of the line (without sound descriptions) per second on screen."""
    spoken = re.sub(r"\(.*?\)", "", re.sub(r"\[.*?\]", "", text))
    return sum(map(count_syllables, re.findall(r"[a-zA-Z]+", spoken))) / duration if duration > 0 else 0


def extract(text: str, duration: float) -> dict:
    """All features this module can compute for one line. The second punctuation count is a column of the thesis table."""
    text = text if isinstance(text, str) else ""
    counts = count_features(text)
    counts["total_number_of_punctuations.1"] = counts.pop("total_number_of_punctuations")
    return {
        **structural_features(text), **lexical_features(text), **politeness_features(text), **counts,
        **tree_features(text), **sluice_features(text), "ellipsis_score": ellipsis_score(text),
        "non_verbal": non_verbal_score(text), "syllables_per_second": syllables_per_second(text, duration),
    }
