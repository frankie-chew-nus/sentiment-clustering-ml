# Preprocessing Decision Record: Airline Alias Audit and Hashtag Suffixes

**Status:** Agreed design; validate against tweet examples before treating alias coverage as final.  
**Module:** Group 1 (preprocessing.py, config.py)  
**Date recorded:** 2026-10-14

## 1. Goal

The project is intended to cluster tweets by overall sentiment, not by which airline is being discussed. Distinctive airline names can introduce airline-identity information into text vectors, so configured airline aliases are normalized to the generic token "airline".

## 2. Dataset-driven findings

A frequency audit was run on the dataset's raw tweet text, without using sentiment labels. The most frequent mentions included:

| Mention | Count | Decision |
|---|---:|---|
| @united | 3,893 | Already covered |
| @usairways | 2,998 | Already covered |
| @americanair | 2,961 | Add alias |
| @southwestair | 2,458 | Already covered |
| @jetblue | 2,248 | Already covered |
| @virginamerica | 517 | Already covered |
| @delta | 68 | Already covered |
| @virginatlantic | 9 | Add as airline alias; normalize to airline |

Frequent hashtags included:

| Hashtag | Count | Decision |
|---|---:|---|
| #jetblue | 48 | Normalize to airline |
| #unitedairlines | 45 | Normalize to airline |
| #usairways | 30 | Normalize to airline |
| #americanairlines | 27 | Normalize to airline |
| #usairwaysfail | 26 | Normalize to airline fail |
| #unitedsucks | 10 | Normalize to airline sucks |

Counts above are from the user's dataset frequency audit. They are descriptive text-frequency counts, not sentiment-label-based analysis.

## 3. Decisions

1. Replace known airline handles with airline.
2. Remove non-airline handles.
3. Replace exact known airline hashtags with airline.
4. For selected airline-prefixed sentiment hashtags, replace the airline prefix with airline and preserve the recognized suffix:
   - #usairwaysfail -> airline fail
   - #unitedsucks -> airline sucks
5. Preserve other hashtag text while dropping only #, e.g. #WorstFlightEver -> WorstFlightEver.
6. Replace configured airline names in ordinary text too, e.g. United delayed my flight -> airline delayed my flight.
7. Keep original raw text in a separate column and do not overwrite it.
8. Keep the dataset sentiment label and confidence out of preprocessing, vectorization, and clustering. Use them only for post-clustering evaluation.

## 4. Why preserve sentiment suffixes?

Removing a whole hashtag such as #usairwaysfail would discard the sentiment-bearing word fail. Converting it to airline fail suppresses the brand name while keeping a potentially useful sentiment signal. The same reasoning applies to #unitedsucks -> airline sucks.

The current implementation uses a configurable suffix list rather than splitting every hashtag with an airline prefix. This is intentionally conservative to reduce false splits, but the suffix list may need expansion after reviewing additional high-frequency hashtags.

## 5. Alias coverage and ambiguity

The aliases are centralized in AIRLINE_ALIASES in config.py. This update adds americanair and virginatlantic, based on observed dataset handles. Note that virginatlantic is not the same brand as Virgin America, but the team explicitly chose to normalize that mention as a generic airline reference too.

Names such as American and Virgin can also be ordinary English words. Whole-word matching reduces partial matches but cannot resolve every semantic ambiguity. Review real before/after examples for false replacements before full-dataset processing.

## 6. Alternatives discussed

- Remove only @mentions and airline hashtags: rejected because ordinary text references such as "United delayed my flight" would retain airline identity.
- Delete airline names: not selected as the default because this can remove useful sentence structure.
- Replace airline names with a generic placeholder: selected because it suppresses brand identity while preserving an entity placeholder.
- Drop the entire airline-prefixed sentiment hashtag: not selected for known suffixes because it would discard sentiment-bearing words.
- Split every concatenated hashtag automatically: not selected here; segmentation tools made incorrect splits in a previous benchmark. The current approach only splits recognized airline prefixes plus configured suffixes.

## 7. Implementation order

1. Remove URLs.
2. Replace configured airline @mentions with airline.
3. Remove remaining non-airline @mentions.
4. Normalize exact airline hashtags and approved airline-prefixed sentiment hashtags.
5. Replace configured airline names in ordinary text.
6. Remove # from other hashtags while retaining content.
7. Normalize whitespace.

## 8. Validation plan and limitations

- Unit-test exact airline names, handles, hashtags, sentiment suffixes, non-airline mentions, whitespace, and invalid input.
- Inspect before/after samples from the actual dataset, especially ambiguous terms.
- Continue to expand aliases and suffixes based on label-free frequency and manual text inspection.
- This remains a curated alias/suffix method, not a general airline entity recognizer.
- Any possible clustering benefit from this normalization must be evaluated later; it is a design hypothesis, not an established performance improvement.


## Group 2: Text normalization (design record)

Group 2 adds independently configurable transforms after Group 1 cleaning. The standard P1 defaults are:

- Expand common contractions and common informal no-apostrophe forms (e.g. can't/cant -> can not; don't/dont -> do not) before punctuation removal.
- Convert standalone numbers to English words (initially ON). Clock times, hyphenated ranges, and alphanumeric identifiers are preserved by the number-token pattern. Number conversion can still change standalone numbers used as identifiers; inspect examples and compare with the setting disabled during ablation.
- Lowercase text (ON). Capitalization features must be calculated from original raw text.
- Punctuation mode is configurable: remove or keep. Removal replaces Unicode punctuation with spaces; emoji symbols are not treated as punctuation by this function.
- Emoji mode is configurable: describe, keep, or remove. P1 starts with describe using the `emoji` library.

Contraction expansion is intentionally conservative for ambiguous forms such as "I'd" and "he's", which are not expanded by the custom mapping. Emoji descriptions may vary with the installed emoji package version. The pipeline requires `emoji` and `num2words` for P1's default describe/number settings; P0 does not call these optional imports.

The order is Group 1 cleanup -> contraction expansion -> emoji handling -> number conversion -> lowercase -> punctuation mode -> whitespace normalization. Contraction expansion precedes punctuation processing so negation is preserved. Feature extraction remains based on raw text. These defaults are hypotheses for evaluation, not evidence that P1 is better than P0.


## Group 3: Advanced lexical normalization (implementation and experiment plan)

Group 3 provides independent switches for word segmentation, SymSpell spelling correction, spaCy lemmatization, and optional stopword removal.

### Decisions and defaults

- Word segmentation supports none, wordninja, symspell, and a conservative hybrid. Default is none because the preliminary benchmark found false splits with both Wordninja and SymSpellPy. SEG0 remains the baseline; other methods are experimental candidates.
- SymSpell spelling correction is optional and OFF by default. It requires a frequency dictionary file supplied via symspell_dictionary_path, uses a limited edit distance, and protects common negation and sentiment/domain terms. Do not enable without specifying and reviewing the dictionary.
- spaCy lemmatization is optional and OFF by default pending an ON/OFF ablation. It requires the en_core_web_sm model. Negation terms are protected by default.
- Stopword removal is optional and OFF by default. When enabled, negation words are preserved by default.
- Hybrid segmentation uses Wordninja candidate splits and wordfreq frequency checks as a heuristic. It is not guaranteed to detect all concatenations correctly.

P2 is intentionally conservative initially: it exposes all Group 3 functions without silently applying unvalidated segmentation or spelling correction. After controlled experiments, choose an evidence-based configuration.

### Dependency setup

Install Python dependencies from requirements.txt and development dependencies from requirements-dev.txt. To use spaCy lemmatization or stopword removal in Colab, also download the English model:

    python -m spacy download en_core_web_sm

For SymSpell segmentation or spelling correction, select a word-frequency dictionary and pass its path in symspell_dictionary_path. Do not commit large dictionaries unless the licence and repository-size implications have been checked.

### Execution order

Group 1 -> contraction expansion -> emoji handling -> number conversion -> lowercasing -> punctuation handling -> segmentation -> spelling correction -> lemmatization -> optional stopword removal -> whitespace normalization.

The original tweet remains unchanged, and engineered structural/VADER features must be computed from raw text. These settings are experimental candidates, not evidence of better clustering.

## Group 1 alias ambiguity trade-off clarified

The team chose to keep the current simple alias-list approach rather than introduce separate context-aware alias systems for handles, hashtags, brands, and locations. The accepted trade-off is that standalone aliases such as "American" or "Virgin" may occasionally be replaced in ordinary-English contexts even when they do not refer to an airline. We will inspect examples and accept this risk for now rather than add more complex entity disambiguation.

Explicit airline names must still be normalized. In particular:

- "American Airlines cancelled my flight" -> "airline cancelled my flight"
- "Virgin America has amazing service" -> "airline has amazing service"

The alias pattern matches whole words/phrases and orders longer aliases before shorter aliases, so the explicit phrase "American Airlines" is treated as a single airline name. This is covered by a regression test. The false-positive risk for standalone words is documented as an accepted limitation, not an intended semantic distinction.

