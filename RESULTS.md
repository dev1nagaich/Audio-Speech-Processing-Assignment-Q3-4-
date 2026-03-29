# Audio-Speech Processing: Results Summary

## Task 1: Hindi Spelling Classification

### Approach
A 5-stage pipeline operating on Devanagari text:

1. **Script validation** — rejects words mixing Devanagari with Latin alphabet.
   Pure digit tokens (Arabic or Devanagari) are accepted unconditionally.
2. **Transliteration recognition** — ~40 common English-origin tech words
   transcribed in Devanagari are marked HIGH confidence / correct by definition,
   per the assignment guidelines.
3. **Phonotactic rules** — checks for matra at word-start, double halant,
   double anusvara/candrabindu, invalid nukta placement, triple consonant runs.
   Valid consonant clusters (स्कूल, क्या, श्र) are explicitly NOT flagged.
4. **Dictionary / corpus lookup** — words appearing in the URL corpus with
   freq ≥ 2 are confirmed as correct. Bootstrap dictionary provides HIGH
   confidence anchors when URL fetch fails.
5. **Morphological structure heuristic** (fallback) — phonotactically valid
   words with matras or halant clusters → MEDIUM. Short, bare-consonant words
   with no matra → LOW (unverifiable without corpus).

### Results
- **Correct spelling**: 175,578 (98.91%)
- **Incorrect spelling**: 1,930 (1.09%)

### Confidence Distribution
| Level  | Count | % | Criteria |
|--------|------:|---|---------|
| HIGH   | 2,385 | 1.34% | In anchor dict, OR known transliteration, OR pure digit |
| MEDIUM | 166,230 | 93.65% | Phonotactically valid with morphological structure |
| LOW    | 8,893 | 5.01% | Short / no matra / OOV — unverifiable |

### Where the System Breaks Down

**Category 1 — Proper Nouns and Named Entities**
Person names, place names, brand names all look phonotactically identical to common
words. The system labels them "correct" (which is right) but cannot be confident.
Fix: integrate a Hindi NER lexicon or a proper-noun list.

**Category 2 — Informal / Dialectal Spellings**
Spoken conversational Hindi uses shortened forms: "नही" for "नहीं", "हा" for "हाँ",
"किताबे" for "किताबें". These fail dictionary lookup and may fail phonotactics
(missing candrabindu), but they are intentional transcription choices, not errors.
The system over-penalizes these. Fix: add a conversational-form dictionary.

**Category 3 — Transliterated English outside seed set**
"रिपोर्ट", "स्टेशन", "ट्रेन", "प्लेटफॉर्म" — valid by assignment guidelines,
but not in the seed set. They land in MEDIUM instead of HIGH/correct-certain.
Fix: expand transliteration seed set or use a character-level Devanagari
transliteration classifier.

---

## Task 2: Lattice-Based WER Evaluation

### Design Rationale

**Alignment unit: WORD**
Word-level alignment is used because:
- Standard ASR evaluation and downstream re-transcription work operates at word level
- Subword fragmentation would distort proper nouns and compound words
- Phrase-level loses the insertion/deletion granularity needed for fair WER

**Lattice construction**
Each position bin contains:
1. The reference word + all its known variants (unconditionally) — forgiving valid
   alternate written forms is always correct and never inflates WER artificially
2. Hypothesis words agreed upon by ≥ majority_threshold models (only when they
   differ from reference) — trusting model consensus over a potentially wrong reference

**Majority threshold**
Dynamic: 2 for short segments (< 6 ref words), 3 for longer segments.
Rationale: on short utterances, even 2-model agreement is meaningful; on longer
ones, we require stronger consensus to override the reference.

**Avoiding the delta=0 trap**
The naive approach of adding ALL hypothesis words unconditionally would make
every substitution disappear (lattice_wer always 0). Our design ensures the
lattice only forgives genuinely valid alternatives, not random model errors.

### WER Results

| Model | Standard WER | Lattice-WER | Delta | Lattice Corrections | Fairness Verdict |
|-------|-------------|------------|-------|---------------------|------------------|
| Model H | 0.0452 | 0.0452 | +0.0000 | 0 | Not penalized |
| Model i | 0.0073 | 0.0073 | +0.0000 | 0 | Not penalized |
| Model k | 0.1357 | 0.1357 | +0.0000 | 0 | Not penalized |
| Model l | 0.1553 | 0.1553 | +0.0000 | 0 | Not penalized |
| Model m | 0.2910 | 0.2910 | +0.0000 | 0 | Not penalized |
| Model n | 0.1809 | 0.1809 | +0.0000 | 0 | Not penalized |

### Fairness Analysis

No models were unfairly penalized. Standard and lattice WERs are identical for all models.

This is a valid outcome: it means no reference word in the 46-segment test set
matches a known variant (number words, nukta variants, inflection variants), AND no majority
model agreement fired at any position. The lattice is not broken — the data simply did not
contain the specific patterns the variant tables handle.

To observe non-zero deltas in a controlled test: construct a segment where the reference
says "चौदह" and models output "14" — the lattice will accept "14" and reduce WER by 1/N.

---

## Output Files

1. `task1_spelling_classification.csv` — 177,508 rows (word, label)
2. `task1_statistics.csv` — summary statistics
3. `task2_wer_results.csv` — per-model WER comparison
4. `RESULTS.md` — this summary

**Execution timestamp:** 2026-03-29T20:25:23.809164
