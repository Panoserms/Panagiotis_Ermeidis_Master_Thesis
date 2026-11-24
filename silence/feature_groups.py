"""The 287 features in the four groups of the thesis (chapter 5)."""

LEXICAL_SEMANTIC_FEATURES = [
    "total_subtlex_us_zipf_of_words", "average_subtlex_us_zipf_of_words_per_word",
    "total_number_of_named_entities", "average_number_of_named_entities_per_word",
    "total_number_of_named_entities_person", "average_number_of_named_entities_person_per_word",
    "simple_type_token_ratio", "simple_type_token_ratio_no_lemma",
    "num_intensifiers", "ratio_intensifiers", "num_hedges", "ratio_hedges",
    "simple_adjectives_variation", "simple_adpositions_variation", "simple_adverbs_variation",
    "simple_auxiliary_variation", "simple_coordinating_conjunctions_variation",
    "simple_determiners_variation", "simple_interjections_variation", "simple_nouns_variation",
    "simple_numerals_variation", "simple_particles_variation", "simple_pronouns_variation",
    "simple_proper_nouns_variation", "simple_punctuations_variation",
    "simple_subordinating_conjunctions_variation", "simple_verbs_variation",
]

STRUCTURAL_FEATURES = [
    "total_number_of_words", "total_number_of_stop_words", "total_number_of_punctuations",
    "total_number_of_syllables", "total_number_of_words_more_than_two_syllables",
    "total_number_of_words_more_than_three_syllables", "total_number_of_unique_words",
    "num_exclamations", "ratio_exclamations", "num_question_marks", "ratio_question_marks",
    "sentence_length", "contrastive", "causal", "additive", "temporal", "conditional",
    "exemplification", "clarification", "adversative",
]

AFFECT_FEATURES = [
    "num_politeness", "num_politeness_single", "num_politeness_phrases", "ratio_politeness",
    "mean_emotion_score", "compound_sentiment_score", "neg", "neu", "pos", "compound",
    "num_positive_terms", "ratio_positive_terms", "num_negative_terms", "ratio_negative_terms",
    "num_emotive_verbs", "ratio_emotive_verbs", "num_emotive_nouns", "ratio_emotive_nouns",
    "num_emotive_adjectives", "ratio_emotive_adjectives",
    "num_emotive_adverbs", "ratio_emotive_adverbs",
    "num_emotive_factive_verbs", "ratio_emotive_factive_verbs",
    "num_positive_adjectives", "ratio_positive_adjectives",
    "num_negative_adjectives", "ratio_negative_adjectives",
    "num_positive_adverbs", "ratio_positive_adverbs",
    "num_negative_adverbs", "ratio_negative_adverbs",
    "num_positive_verbs", "ratio_positive_verbs",
    "num_negative_verbs", "ratio_negative_verbs",
    "num_positive_nouns", "ratio_positive_nouns",
    "num_negative_nouns", "ratio_negative_nouns",
    "neutral", "approval", "annoyance", "realization", "confusion", "anger",
    "disapproval", "disappointment", "curiosity", "amusement", "sadness",
    "excitement", "optimism", "joy", "caring", "disgust", "admiration",
    "desire", "fear", "surprise", "love", "gratitude", "embarrassment",
    "nervousness", "grief", "remorse", "relief", "pride",
]

SYNTACTIC_FEATURES = [
    # POS counts
    "total_number_of_adjectives", "total_number_of_unique_adjectives", "average_number_of_adjectives_per_word",
    "total_number_of_adpositions", "total_number_of_unique_adpositions", "average_number_of_adpositions_per_word",
    "total_number_of_adverbs", "total_number_of_unique_adverbs", "average_number_of_adverbs_per_word",
    "total_number_of_auxiliaries", "total_number_of_unique_auxiliaries", "average_number_of_auxiliaries_per_word",
    "total_number_of_coordinating_conjunctions", "total_number_of_unique_coordinating_conjunctions", "average_number_of_coordinating_conjunctions_per_word",
    "total_number_of_determiners", "total_number_of_unique_determiners", "average_number_of_determiners_per_word",
    "total_number_of_interjections", "total_number_of_unique_interjections", "average_number_of_interjections_per_word",
    "total_number_of_nouns", "total_number_of_unique_nouns", "average_number_of_nouns_per_word",
    "total_number_of_numerals", "total_number_of_unique_numerals", "average_number_of_numerals_per_word",
    "total_number_of_particles", "total_number_of_unique_particles", "average_number_of_particles_per_word",
    "total_number_of_pronouns", "total_number_of_unique_pronouns", "average_number_of_pronouns_per_word",
    "total_number_of_proper_nouns", "total_number_of_unique_proper_nouns", "average_number_of_proper_nouns_per_word",
    "total_number_of_punctuations.1", "total_number_of_unique_punctuations", "average_number_of_punctuations_per_word",
    "total_number_of_subordinating_conjunctions", "total_number_of_unique_subordinating_conjunctions", "average_number_of_subordinating_conjunctions_per_word",
    "total_number_of_verbs", "total_number_of_unique_verbs", "average_number_of_verbs_per_word",
    # Dependency relations
    "ROOT", "nsubj", "ccomp", "punct", "dobj", "prt", "acl", "acomp",
    "advcl", "advmod", "agent", "amod", "appos", "attr", "aux", "auxpass",
    "case", "cc", "conj", "csubj", "csubjpass", "dative", "dep", "det",
    "expl", "intj", "mark", "meta", "nmod", "npadvmod", "nsubjpass",
    "nummod", "oprd", "parataxis", "pcomp", "pobj", "poss", "preconj",
    "predet", "prep", "quantmod", "relcl", "xcomp",
    # Tree structure
    "tree_depth_max", "tree_depth_mean", "branching_factor_mean", "branching_factor_max",
    "n_embedded_clauses", "left_branching_count", "right_branching_count", "left_branching_ratio",
    # Dependency distances
    "dep_distance_mean", "dep_distance_max", "dep_distance_sum", "dep_distance_std",
    "dep_distance_amod_mean", "dep_distance_amod_max",
    "dep_distance_nsubj_mean", "dep_distance_nsubj_max",
    "dep_distance_dobj_mean", "dep_distance_dobj_max",
    "dep_distance_compound_mean", "dep_distance_compound_max",
    "dep_distance_advcl_mean", "dep_distance_advcl_max",
    "dep_distance_prep_mean", "dep_distance_prep_max",
    "dep_distance_nsubjpass_mean", "dep_distance_nsubjpass_max",
    "dep_distance_relcl_mean", "dep_distance_relcl_max",
    # Passive / relative clauses
    "has_passive", "passive_clause_count", "passive_ratio",
    "relcl_in_subject_np", "has_relcl_in_subject",
    # Subject NP complexity
    "subject_np_length", "subject_np_depth", "subject_has_pp", "subject_has_relcl",
    # Clause counts
    "n_main_clauses", "n_coordinate_clauses", "n_subordinate_clauses", "total_clauses",
    "main_clause_ratio", "coordinate_clause_ratio", "subordinate_clause_ratio",
    # Phrase-level
    "ends_at_clause_boundary", "n_noun_phrases", "np_length_mean", "np_length_max",
    "n_verb_phrases", "vp_size_mean", "vp_size_max", "n_nested_pps",
    # Completeness / constructions
    "ends_syntactically_complete", "ends_with_open_dep",
    "has_fronted_adverbial", "has_existential", "has_cleft", "has_topicalization", "n_non_canonical",
    # Final-position
    "final_pos_content_word", "final_pos_nominal", "final_pos_verbal",
    "final_pos_function_word", "final_pos_connector", "final_pos_discourse",
    "final_dep_completing", "final_dep_clausal", "final_dep_open", "final_dep_subject", "final_dep_modifier",
    # Bigram patterns
    "bigram_verb_to_nominal", "bigram_ends_nominal", "bigram_ends_verbal",
    "bigram_ends_function", "bigram_nominal_to_nominal", "bigram_modifier_to_modifier",
    "bigram_single_word", "bigram_ends_intj",
    # Turn-final
    "is_bare_wh", "ends_on_cconj", "aposiopesis_score", "non_verbal",
    # Ellipsis
    "ellipsis_score",
    # Prosodic rate
    "syllables_per_second",
]

GROUPS = {
    "Syntactic": SYNTACTIC_FEATURES,
    "Lexical-semantic": LEXICAL_SEMANTIC_FEATURES,
    "Structural": STRUCTURAL_FEATURES,
    "Affect": AFFECT_FEATURES,
}

# Yes/no features: kept as 0/1, not standardised.
BOOLEAN_FEATURES = [
    "has_passive", "has_relcl_in_subject", "subject_has_pp", "subject_has_relcl",
    "ends_at_clause_boundary", "ends_syntactically_complete", "ends_with_open_dep",
    "has_fronted_adverbial", "has_existential", "has_cleft", "has_topicalization",
    "final_pos_content_word", "final_pos_nominal", "final_pos_verbal",
    "final_pos_function_word", "final_pos_connector", "final_pos_discourse",
    "final_dep_completing", "final_dep_clausal", "final_dep_open",
    "final_dep_subject", "final_dep_modifier",
    "bigram_verb_to_nominal", "bigram_ends_nominal", "bigram_ends_verbal",
    "bigram_ends_function", "bigram_nominal_to_nominal", "bigram_modifier_to_modifier",
    "bigram_single_word", "bigram_ends_intj",
    "is_bare_wh", "ends_on_cconj",
]
