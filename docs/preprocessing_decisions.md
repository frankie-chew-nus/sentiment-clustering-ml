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

## Group 1 real-dataset audit (Colab)

**Audit method:** A reproducible sample of 100 tweets was processed using `P0_CONFIG`, and a separate sample containing the terms "American" or "Virgin" was inspected. The audit used tweet text only and did not use sentiment labels.

### Results that matched the intended rules

- The 100-tweet sample reported **100 changed, 0 unchanged**. This is consistent with a sample drawn from a dataset dominated by airline @mentions; it is not an accuracy score and does not imply every transformation was correct.
- Common handles such as `@SouthwestAir`, `@AmericanAir`, `@JetBlue`, and `@USAirways` were replaced with `airline`.
- The following targeted hashtag checks produced the expected output:
  - `#United` -> `airline`
  - `#WorstFlightEver` -> `WorstFlightEver`
  - `#usairwaysfail` -> `airline fail`
  - `#unitedsucks` -> `airline sucks`
  - `#badcustomerservice` -> `badcustomerservice` (Group 1 removes the hash marker; general segmentation is not part of Group 1)

### Known false replacements observed and accepted

The ambiguous-term sample showed examples including:

- `Virgin Islands` -> `airline Island`
- `Virgin Mobile` -> `airline mobile`
- Ordinary uses of `American` may be replaced with `airline`

These are false replacements when the word does not refer to an airline. The team explicitly chose to accept this trade-off for now in order to keep alias handling simple, rather than add context-aware entity disambiguation. This limitation should remain visible in the methodology and can be revisited if it materially affects later clustering results.

### Group 1 audit conclusion

Group 1's tested behavior matches the team's current rules, including generic airline placeholders and preservation of the suffix for the two agreed airline sentiment hashtags. Unit tests and this small manual audit are useful validation, but they do not establish that every airline alias or every tweet is handled correctly. The alias list remains curated, and the ambiguity trade-off is accepted rather than resolved.

**Next step:** Proceed to Group 2 normalization while retaining raw tweet text separately. Keep this audit record as the rationale for the current Group 1 behavior.

## Group 2 validation: revised P1 defaults

Manual Colab validation was performed one operation at a time and then on the combined P1 pipeline.

### Contractions

The following examples expanded as intended and preserved explicit negation: can't -> can not, won't -> will not, don't -> do not, isn't -> is not, haven't -> have not. Informal no-apostrophe forms cant and dont were also expanded in the tested examples.

### Emoji handling

Describe, keep, and remove modes were checked. Describe mode preserved semantic text such as:
- angry emoji -> "enraged face"
- smiling emoji -> "smiling face with smiling eyes"
- red heart -> "red heart"
- crying emoji -> "loudly crying face"
- thumbs-up emoji -> "thumbs up"

P1 keeps describe mode as the default.

### Lowercasing

Uppercase text was normalized successfully while punctuation and emojis remained available to their own processing steps. Capitalization-based engineered features must still be calculated from raw text.

### Numbers: decision changed to KEEP by default

Generic number-to-word conversion worked for quantities such as "3 hours" -> "three hours", but it also converted identifiers such as "gate 25" and "flight 123". Structured expressions such as times, ranges, ratings, and alphanumeric identifiers can carry structure that should not be destroyed.

The team therefore changed P1's default from number conversion ON to OFF. Number conversion remains available as a configurable experiment. Examples intentionally preserved by the revised P1 include 10/10, -9/10, 1:30, 1-4am, N351JB, gate 25, and flight 123.

### Punctuation: decision changed to KEEP by default

Blanket punctuation removal produced destructive transformations including 10/10 -> "10 10" and -9/10 -> "9 10". While contraction expansion protects negation before punctuation handling, ratings and character-level patterns remain vulnerable.

The team therefore changed P1's punctuation mode from remove to keep. Word TF-IDF can apply its own tokenization, while character TF-IDF can retain potentially useful patterns such as !!!, ?!?, 10/10, and -9/10. Punctuation-derived structural features remain calculated from raw text.

### Combined revised P1 manual verification

The temporary revised configuration (contractions ON, emoji describe, numbers keep, lowercase ON, punctuation keep) produced the intended examples:
- "I can't believe this!!!" -> "i can not believe this!!!"
- "Terrible!!! [angry emoji]" -> "terrible!!! enraged face"
- "10/10 service [smiling emoji]" -> "10/10 service smiling face with smiling eyes"
- "-9/10 experience" -> "-9/10 experience"
- "Waited 3 hours!!!" -> "waited 3 hours!!!"
- "Flight N351JB delayed AGAIN?!" -> "flight n351jb delayed again?!"
- "@AmericanAir I won't fly American Airlines again!!! [angry emoji]" -> "airline i will not fly airline again!!! enraged face"
- "@USAirways #usairwaysfail" -> "airline airline fail"

These examples are manual functional checks, not evidence that P1 improves clustering. P0/P1 and optional preprocessing choices will be compared later under the experiment design.

## Group 3 configuration refactor: protected lexical terms

The project separates experimental choices from processing logic. Negation terms, sentiment/domain terms, intensifiers, and the combined stopword-protection set now live in config.py rather than being hard-coded inside lexical_normalization.py.

The current categories are:
- NEGATION_WORDS: terms such as no, not, never and common negative contractions.
- SENTIMENT_PROTECTED_WORDS: airline-domain and sentiment-bearing terms such as delay, cancel, refund, terrible, good, worst, happy, angry, love and hate.
- INTENSIFIER_WORDS: candidate modifiers such as very, really, so, too, extremely, absolutely and seriously.
- STOPWORD_PROTECTED_WORDS: union of the above sets for stopword-removal experiments.

These are configurable research choices, not claims that the lists are complete or optimal. They can be revised from label-free corpus analysis and manual review without changing the processing functions. The stopword function also accepts additional protected_words for experiment-specific overrides.

## G3.3 corpus audit: hashtag-focused word segmentation

A label-free corpus audit found 2,006 unique hashtags. Among the most frequent hashtags, many sentiment-bearing expressions were intentionally concatenated, including `neveragain`, `badservice`, `badcustomerservice`, `nothappy`, `notcool`, `worstairlineever`, `worstcustomerservice`, `poorcustomerservice`, `customerservicefail`, and `greatservice`.

The diagnostic compared `none`, Wordninja, and the current hybrid method on 63 frequent hashtag candidates of length >= 7. At least one method changed 35 candidates. In this sample, Wordninja and the current hybrid produced the same displayed outputs, so the hybrid did not demonstrate an advantage over Wordninja in this audit.

Useful examples included `neveragain` -> `never again`, `badcustomerservice` -> `bad customer service`, `nothappy` -> `not happy`, `notcool` -> `not cool`, `worstcustomerservice` -> `worst customer service`, `poorcustomerservice` -> `poor customer service`, `customerservicefail` -> `customer service fail`, and `greatservice` -> `great service`.

False or low-quality splits were also observed, including `custserv` -> `cus tse rv` and `brandmance` -> `brand man ce`.

### Interpretation and recommendation

Hashtags are more likely than ordinary long words to contain intentional concatenation because multi-word hashtag phrases are conventionally written without spaces. The audit therefore supports a narrower use of segmentation: apply Wordninja selectively to hashtag-derived compound tokens rather than indiscriminately to all long tokens in ordinary tweet text.

Airline-specific hashtag normalization should take precedence over generic segmentation. For example, the existing rules `#usairwaysfail` -> `airline fail` and `#unitedsucks` -> `airline sucks` are preferable to allowing generic segmentation to reintroduce airline names.

Recommended conceptual order: (1) detect/extract the hashtag while provenance is known; (2) apply airline-specific hashtag normalization first; (3) for remaining non-airline hashtags, optionally segment the hashtag body with Wordninja; (4) continue with the normal text pipeline and later lemmatization; (5) do not apply Wordninja globally to ordinary long words unless a separate experiment supports it.

The current Group 1 implementation strips `#` before Group 3, which loses hashtag provenance. Before enabling selective hashtag segmentation, the architecture should be revised so hashtag provenance survives until the hashtag-specific segmentation decision has been made. This finding is based on unlabeled corpus structure and manual inspection, not sentiment labels.

### Ordinary-text segmentation audit

To test the trade-off of hashtag-only segmentation, hashtags, mentions, and URLs were excluded and long ordinary-text tokens (length >= 7) were audited separately. The corpus contained **4,677 unique long ordinary-text tokens**. Wordninja proposed a split for **562** of them.

High-frequency problematic examples included:

| Ordinary token | Count | Wordninja output | Assessment |
|---|---:|---|---|
| `flightled` | 508 | `flight led` | Incorrect interpretation; this is a noisy-token/correction problem, not a compound-segmentation problem |
| `flighted` | 228 | `flight ed` | Incorrect split |
| `flightr` | 162 | `flight r` | Incorrect split |
| `rebooked` | 139 | `re booked` | Damages a valid word |
| `reflight` | 52 | `re flight` | Questionable and not sufficient evidence for global segmentation |

Lower-frequency false splits also included `cinnabons` -> `cinnabon s` and `prefundia` -> `pre fund i a`.

This provides corpus-specific evidence against global Wordninja segmentation of ordinary tweet text. The final G3.3 recommendation is therefore: **enable Wordninja only for non-airline hashtag bodies in the P2 candidate; keep global ordinary-text segmentation OFF.** Noisy ordinary tokens such as `flightled`, `flighted`, and `flightr` should be handled, if at all, by a separate spelling/noise-normalization decision rather than by segmentation.

## G3.4 investigation: malformed flight tokens and spelling-correction objective

During the ordinary-text segmentation audit, a family of frequent malformed `flight*` tokens was discovered. A raw-text audit (no sentiment labels) found 46 flight-related token variants. High-frequency suspicious forms included `flightled` (508 token occurrences in the token audit), `flighted` (228), `flightr` (162), and `reflight` (53), alongside additional lower-frequency variants.

Context inspection showed these are systematic corpus artifacts rather than simple intentional compounds. Repeated contexts included constructions such as `Cancelled Flightled flight`, `Cancelled Flighted flight`, `hours Late Flightr`, and `reFlight Booking Problems`. This explains why global Wordninja segmentation was inappropriate for them: it produced outputs such as `flight led`, `flight ed`, and `flight r`, which solve the wrong problem.

### Sentiment-focused interpretation

The project objective is sentiment clustering, not grammatical reconstruction of every tweet. In the observed contexts, the principal sentiment-bearing information usually remains in surrounding terms such as `cancelled`, `late`, `problems`, `delayed`, `terrible`, etc. The malformed tokens themselves primarily create vocabulary fragmentation/noise for word TF-IDF. Leaving them unchanged is therefore possible, but it creates extra artificial vocabulary dimensions; manually mapping each artifact could reduce that noise but would introduce dataset-specific handcrafted rules.

### Decision: avoid handcrafted artifact mappings and evaluate generic correction

The team does not want to maintain a handcrafted correction dictionary for these artifacts. Instead, return to the original preprocessing objective: determine whether a general spelling/noise-correction method can reduce vocabulary fragmentation while preserving sentiment-bearing terms.

SymSpell will therefore be evaluated empirically on the actual unlabeled corpus vocabulary. The question is not whether it perfectly reconstructs the original English, but whether its corrections are useful for sentiment-oriented vectorization without damaging valid airline-domain, sentiment, abbreviation, airport, name, or Twitter vocabulary.

Evaluation should classify changed tokens as useful correction, harmless normalization, or harmful correction and should consider token frequency so that high-frequency errors matter more than one-off errors. Existing configurable protected sentiment/domain and negation terms remain safeguards. No sentiment labels or sentiment-confidence values are to be used in this investigation.

No artifact-specific code change is made at this stage. The next G3.4 experiment is a corpus-level SymSpell correction audit; spelling correction will be enabled in P2 only if that audit provides sufficient evidence.

## G3.4 SymSpell corpus audit and decision

Following the malformed-token investigation, generic SymSpell correction was evaluated against the unlabeled ordinary-text vocabulary rather than using handcrafted artifact mappings. The standard `symspellpy` English frequency dictionary `frequency_dictionary_en_82_765.txt` was used with maximum edit distance 2. Sentiment labels and sentiment-confidence values were not used.

### Small diagnostic

The initial diagnostic showed mixed behavior. Useful changes included `flightr` -> `flight`, `terribl` -> `terrible`, and `disapointed` -> `disappointed`. Important harmful changes included `reflight` -> `preflight` and `rebooked` -> `rebooted`. `flightled` became `flighted`, while `flighted` remained unchanged, so SymSpell did not reliably solve the systematic malformed `flight*` family.

### Corpus-wide audit

To focus on changes with material corpus impact, ordinary-text vocabulary terms occurring at least 5 times were audited.

- Frequent vocabulary terms audited: **2,852**
- Unique terms SymSpell would change: **77 / 2,852 (2.7%)**
- Audited token occurrences: **224,765**
- Token occurrences affected by proposed corrections: **2,723 / 224,765 (1.2%)**

Although the overall change rate was low, several high-frequency changes were harmful for sentiment or airline/Twitter meaning. Examples included:

| Original | SymSpell | Count | Concern |
|---|---|---:|---|
| `flightled` | `flighted` | 508 | Does not resolve the systematic artifact |
| `didn` | `did` | 202 | Can destroy negation intent |
| `doesn` | `does` | 169 | Can destroy negation intent |
| `flightr` | `flight` | 162 | Useful correction |
| `fleek` | `fleet` | 150 | Changes meaning |
| `rebooked` | `rebooted` | 139 | Damages valid airline-domain vocabulary |
| `mins` | `mind` | 127 | Damages a common time abbreviation |
| `couldn` | `could` | 96 | Can destroy negation intent |
| `wasn` | `was` | 90 | Can destroy negation intent |
| `cust` | `just` | 55 | Damages an airline/customer-service abbreviation |
| `wouldn` | `would` | 53 | Can destroy negation intent |
| `reflight` | `preflight` | 52 | Changes meaning |
| `aren` | `are` | 48 | Can destroy negation intent |
| `flighting` | `lighting` | 44 | Changes meaning |
| `checkin` | `checking` | 38 | Damages airline-domain vocabulary |
| `shouldn` | `should` | 35 | Can destroy negation intent |
| `yall` | `all` | 32 | Damages informal Twitter language |
| `hasn` | `has` | 30 | Can destroy negation intent |
| `haha` | `hama` | 30 | Introduces unrelated vocabulary |
| `airplane` | `airline` | 22 | Changes meaning |
| `onboard` | `inboard` | 21 | Changes meaning |
| `tweeting` | `sweeting` | 20 | Damages Twitter vocabulary |
| `tweets` | `sweets` | 17 | Damages Twitter vocabulary |
| `dmed` | `died` | 15 | Changes social-media meaning and may introduce false sentiment |
| `deplane` | `replace` | 13 | Damages airline-domain vocabulary |
| `flightation` | `flirtation` | 13 | Introduces unrelated vocabulary |
| `trueblue` | `treble` | 11 | Damages domain/brand vocabulary |
| `flts` | `fits` | 10 | Damages flight abbreviation |
| `deplaned` | `replaced` | 8 | Damages airline-domain vocabulary |
| `resched` | `reached` | 7 | Damages rescheduling abbreviation |
| `assult` | `assault` | 5 | Useful correction |

Additional potentially harmful corrections included `tmrw` -> `term`, `intl` -> `into`, `bday` -> `day`, `thnx` -> `the`, `lmao` -> `lao`, `carseat` -> `carpet`, and `vieques` -> `venues`.

### Sentiment-oriented interpretation

The key criterion is not perfect spelling or grammatical reconstruction; it is whether preprocessing reduces vocabulary fragmentation without damaging sentiment-bearing context. SymSpell changed only 1.2% of audited token occurrences, but some of those changes affected high-frequency negation fragments and valid airline/Twitter vocabulary. In particular, corrections such as `didn` -> `did`, `doesn` -> `does`, `couldn` -> `could`, and similar forms risk weakening polarity information, while corrections such as `rebooked` -> `rebooted` introduce unrelated concepts.

### Decision

**Generic SymSpell spelling correction remains OFF for P2.** The potential benefit from genuine typo correction does not justify the observed semantic and sentiment risks in this corpus. The team will not add a large handcrafted exception list merely to make generic SymSpell safe, because that would undermine the goal of a general, reproducible preprocessing method. The systematic malformed `flight*` artifacts are acknowledged as corpus noise and are not given handcrafted mappings at this stage.

## P2 final lexical-normalization configuration

After the Group 3 investigations, P2 now enables spaCy lemmatization and stopword removal with the existing configured protection lists. This follows the sentiment-clustering objective: collapse morphological variants while removing low-information grammatical vocabulary, but retain negation, sentiment/domain terms, and configured intensifiers.

Manual lemmatization checks previously showed useful normalization such as `flights` -> `flight`, `delayed`/`delaying` -> `delay`, `cancelled` -> `cancel`, `disappointed` -> `disappoint`, and `responded` -> `respond`, while `not` and `never` were preserved. Forms of `be` may collapse (`was`/`is`/`am`/`are` -> `be`), reducing grammatical vocabulary that is not central to the sentiment objective.

Manual stopword checks showed the intended protected behavior: `the flight was delayed` -> `flight delayed`; `I am not satisfied` -> `not satisfied`; `I will never fly again` -> `never fly`; `the service was very good` -> `service very good`; `I am really disappointed with the service` -> `really disappointed service`; and `this is extremely bad` -> `extremely bad`.

Final P2 Group 3 defaults are therefore: selective Wordninja segmentation for non-airline hashtag bodies ON; global ordinary-text segmentation OFF; generic SymSpell spelling correction OFF; lemmatization ON; stopword removal ON; negation protection ON. Stopword removal uses the configured protection sets for negation, sentiment/domain terms, and intensifiers. These preprocessing choices remain hypotheses whose downstream value will be tested by the clustering experiments; functional validation does not by itself establish improved clustering quality.


## P0 update: decode HTML entities before URL removal

Raw tweet inspection identified HTML character references such as `&amp;`, which represent encoded text rather than intended linguistic content. P0 now uses Python's built-in `html.unescape()` to decode HTML character references before other cleaning. No additional dependency is required.

The operation runs **before URL removal**. This restores the serialized text first (for example, `&amp;` -> `&`, `&gt;` -> `>`), after which URL removal and the remaining P0 rules operate on the decoded representation. If an HTML entity occurs inside a URL, decoding it first is harmless because the subsequent URL-removal step removes the URL as a whole.

This is treated as low-level text restoration rather than lexical normalization: it does not intentionally infer sentiment or rewrite vocabulary. The setting is configurable through `decode_html_entities` and is enabled in P0, therefore inherited by P1 and P2. As with the rest of the pipeline, the original raw text remains retained separately.
