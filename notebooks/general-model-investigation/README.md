# General-Model Investigation — Qwen Preprocessing Ablation (E24–E26)

## Purpose

This independent follow-up investigates **whether traditional text preprocessing helps an instruction-tuned general-purpose language model classify airline sentiment**. It builds on [E23 — Qwen context-aware question interpretation](../model%20ablation/L6_E23_Qwen_ContextAwareQuestions.ipynb), which used original redacted tweet text and achieved the strongest exploratory Macro F1 among the earlier prompt experiments.

**Research question:** Holding the model, sentiment prompt, inference settings, dataset and scoring procedure constant, what happens when the input tweets are processed with P0, P1 or P2 instead of being passed as raw text?

The study uses **14,639 redacted airline tweets**. No additional training or fine-tuning is performed. This is **direct zero-shot classification**, not K-means, GMM or Agglomerative Clustering.

## Experimental design

| Experiment | Input | Preparation | What changes? |
|---|---|---|---|
| **E23 — comparison baseline** | RAW | Original redacted tweet text | No preprocessing |
| **E24** | P0 | Basic cleaning: decode HTML entities; remove URLs/non-airline mentions; normalize airline names, handles and hashtags; whitespace cleanup | Basic cleanup |
| **E25** | P1 | P0 plus contraction expansion, lowercasing and emoji descriptions; retain numbers and punctuation | Text normalization |
| **E26** | P2 | P1 plus selective non-airline hashtag segmentation, spaCy lemmatization and stopword removal with negation protection | Advanced lexical normalization |

P0/P1/P2 are regenerated from the repository's reusable `preprocessing.py` and `config.py` definitions, **not loaded from earlier processed CSVs or feature matrices**. P2 keeps global segmentation and generic spelling correction disabled. These settings reflect the repository's current configuration; they are not separate sentiment models.

**Controlled factors:** All four runs use `Qwen/Qwen2.5-3B-Instruct`, E23's same context-aware question prompt, deterministic greedy decoding, the same 14,639-row redacted input corpus, and the same label-alignment and scoring protocol. The only intended experimental change is the text supplied to the model.

**Label boundary:** Predictions are generated and frozen before the original human sentiment annotations are loaded for evaluation. No labels are supplied as few-shot examples. The experiment notebooks use distinct output directories and resumable checkpoints.

## Notebooks

- [E24 — Qwen + P0](L6_E24_Qwen_P0_Preprocessing_Standalone.ipynb)
- [E25 — Qwen + P1](L6_E25_Qwen_P1_Preprocessing_Standalone.ipynb)
- [E26 — Qwen + P2](L6_E26_Qwen_P2_Preprocessing_Standalone.ipynb)
- [E23 — RAW comparison baseline](../model%20ablation/L6_E23_Qwen_ContextAwareQuestions.ipynb)

These repository notebook copies are **clean, unexecuted Colab notebooks**. The values below were extracted from the separate user-executed notebooks and their recorded outputs. This README is a summary, not a substitute for retaining executed notebooks, manifests, metrics JSON and frozen predictions.

## Results

| Experiment | ARI ↑ | NMI ↑ | Purity ↑ | Direct accuracy ↑ | Macro F1 ↑ | Silhouette |
|---|---:|---:|---:|---:|---:|---|
| **E23 — RAW** | **0.5873** | **0.4347** | **0.8230** | **82.30%** | **0.7603** | N/A |
| E24 — P0 | 0.5780 | 0.4189 | 0.8219 | 82.19% | 0.7598 | N/A |
| E25 — P1 | 0.5765 | 0.4185 | 0.8223 | 82.23% | 0.7582 | N/A |
| E26 — P2 | 0.4461 | 0.2973 | 0.7641 | 76.41% | 0.6955 | N/A |

**Accuracy** is direct sentiment-classification accuracy, not Hungarian-mapped cluster accuracy. **Purity** is descriptive for predicted sentiment groups; its equality with accuracy here is specific to these runs. **Silhouette** is not applicable to direct classification.

### Per-class F1

| Experiment | Negative F1 ↑ | Neutral F1 ↑ | Positive F1 ↑ |
|---|---:|---:|---:|
| E23 — RAW | **0.9029** | 0.5964 | **0.7818** |
| E24 — P0 | 0.9003 | **0.6020** | 0.7770 |
| E25 — P1 | 0.8999 | 0.5946 | 0.7799 |
| E26 — P2 | 0.8573 | 0.5283 | 0.7011 |

### Neutral recall

| Experiment | Neutral recall | Correctly classified neutral tweets (of 3,099) |
|---|---:|---:|
| E23 — RAW | 52.92% | 1,640 |
| E24 — P0 | **54.37%** | **1,685** |
| E25 — P1 | 52.27% | 1,620 |
| E26 — P2 | 51.69% | 1,602 |

## Findings and interpretation

### 1. RAW, P0 and P1 are practically close

The difference in Macro F1 between RAW (0.7603), P0 (0.7598) and P1 (0.7582) is small. The data **do not demonstrate a substantial overall benefit** from applying P0 or P1 to Qwen. RAW is marginally ahead on aggregate metrics, but these differences have not been tested for statistical significance.

### 2. P0 offers a modest neutral-class trade-off

P0 identified **45 more human-labelled neutral tweets** than RAW and achieved a slightly higher neutral F1 (0.6020 versus 0.5964). This came with a reduction in positive recall (approximately 89.25% for RAW versus 83.36% for P0). Better neutral recovery therefore did not yield a clear overall improvement.

### 3. P2 substantially reduced sentiment recovery

Compared with RAW, P2 reduced accuracy by about **5.89 percentage points** (82.30% to 76.41%) and Macro F1 by about **0.0648** (0.7603 to 0.6955). Negative, neutral and positive F1 all declined. This is the clearest performance difference in the preprocessing ablation.

A plausible mechanism is **loss of natural-language context**: lemmatization and stopword removal can change sentence structure, remove grammatical cues and weaken the relationships an instruction-following model uses to infer the author's attitude. This is a **hypothesis**, not a causal finding; P2 changes several operations together, so we have not isolated which one is responsible.

### 4. Preprocessing is representation- and task-dependent

P0/P1/P2 were designed as modular cleaning options for statistical NLP and feature engineering. Those motivations do not guarantee that the same transformations benefit a pretrained instruction model that was designed to interpret natural sentences.

**Main conclusion:** More aggressive preprocessing is not automatically better. For the tested Qwen zero-shot sentiment task, original text and lightly normalized text performed similarly, while advanced P2 lexical normalization substantially underperformed.

## Limitations and next steps

- **Exploratory design:** E24–E26 were proposed after examining E19–E23 results on this same labelled corpus. The comparisons are descriptive and should be validated on a separate untouched test set or newly annotated tweets.
- **No causal decomposition:** P2 combines hashtag segmentation, lemmatization and stopword removal. A follow-up component ablation would be needed to identify the damaging operation(s).
- **Model and domain specificity:** These results concern one Qwen model and one airline-tweet corpus. They do not show that P2 is universally harmful or that RAW always wins.
- **Reproducibility:** Preserve model/prompt identifiers, preprocessing configuration, source hashes, row order, generation settings, software versions, checkpoint identities, predictions and evaluation artifacts. Fresh runs on different Colab package versions are not guaranteed to reproduce every prediction exactly.
- **Privacy:** Do not commit original labelled data, secrets or tweet-level prediction checkpoints to the public repository.

**Recommendation:** Keep **E23 RAW** as the preferred overall configuration for now. Treat **E24 P0** as a candidate when neutral F1 is specifically prioritized, and avoid E26 P2 for this tested setup unless a component ablation reveals a useful subset of its operations.
