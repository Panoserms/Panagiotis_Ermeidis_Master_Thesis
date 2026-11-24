# Data

| File | What it is |
|---|---|
| `features_train.csv.gz` | 10,376 pairs: film, gap in seconds, label, 287 features |
| `features_validation.csv.gz` | 1,297 pairs |
| `features_test.csv.gz` | 1,297 pairs |
| `predictions_bert_linear.csv` | test set: label, prediction and score of the linear SVM on BERT-Large embeddings |
| `predictions_mistral_linear.csv` | the same for Mistral-7B |

Rows are in the same order in `features_test.csv.gz` and the two prediction files. Label 1 = the pause after the line is 700 ms or longer.
The features are not standardised yet; `silence.models.load_features` does that with the training set.

## What is not here

**Subtitles.** The film dialogue is copyrighted, so the tables hold numbers only. To run the marked cells of the notebook, put the `.srt`
files of these films into `data/subtitles/` under these names (from [OpenSubtitles](https://www.opensubtitles.org)). A different subtitle
version of a film gives slightly different pairs.

| File | Film |
|---|---|
| `basis.srt` | On the Basis of Sex |
| `benefits.srt` | Friends with Benefits |
| `conviction.srt` | Conviction |
| `crazystupid.srt` | Crazy, Stupid, Love |
| `hacksaw.srt` | Hacksaw Ridge |
| `harriet.srt` | Harriet |
| `help.srt` | The Help |
| `hidden.srt` | Hidden Figures |
| `immitation.srt` | The Imitation Game |
| `king.srt` | King Richard |
| `pslove.srt` | P.S. I Love You |

Thirteen files were collected. *The Six Triple Eight* (`6888.srt`) and *Oppenheimer* were left out because of data problems.

**Embeddings.** The two scripts in `gpu/` make them from `data/pairs/`, which section 1 of the notebook writes. Put the results into
`data/embeddings/`.

**Lexicons and 76 features.** The NRC Emotion Lexicon may not be passed on. The code for the affect features (except politeness), the
discourse connectives, hedges and intensifiers is not part of this repository; their values are in the tables.
