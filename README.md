# Audio Speech Processing Assignment (Q3 and Q4)

This repository contains the assignment solution with the required mapping:
- Task 1 corresponds to Question 3
- Task 2 corresponds to Question 4

## Task 1 (Question 3): Hindi Unique Word Spelling Classification

Objective:
Classify unique Hindi words into:
- correct spelling
- incorrect spelling

Implemented approach (from task_complete.py):
1. Script validation
- Rejects mixed Devanagari and Latin words
- Accepts numeric tokens (Arabic/Devanagari digits)

2. Transliteration handling
- Recognizes common English-origin words written in Devanagari
- Treats these as valid, per assignment guideline

3. Phonotactic checks
- Detects invalid Hindi orthographic patterns (for example invalid matra placement, repeated markers, invalid nukta placement)

4. Corpus/dictionary support
- Builds frequency evidence from transcription URLs
- Uses dictionary/frequency thresholds for stronger decisions

5. Fallback morphological heuristic
- Used when URL-based evidence is weak
- Assigns confidence based on structural plausibility

Output per word:
- label: correct spelling or incorrect spelling
- confidence: HIGH, MEDIUM, LOW
- reason: short explanation for the decision

Low-confidence review and reliability analysis:
- The script samples low-confidence words and reports bucket behavior
- It also documents failure categories where the system is less reliable (for example named entities, informal spellings, unseen transliterations)

Final count required in deliverable:
- Unique correct spelled words = 175,578

## Task 2 (Question 4): Lattice-Based WER Evaluation

Objective:
Compute fairer ASR WER by replacing rigid single-reference matching with lattice-based matching.

Design used:
1. Alignment unit
- Word-level alignment (standard for WER and segment-level transcription evaluation)

2. Lattice construction
- Each reference position is a bin
- Bin includes:
  - reference token and known valid variants
  - model-agreed alternatives (majority-based) when disagreement suggests reference may be wrong

3. Error handling philosophy
- Insertions/deletions remain penalized
- Substitutions are forgiven only when hypothesis token is inside that bin's valid alternatives

4. Trusting models over reference
- Uses dynamic majority thresholds across segment lengths
- Prevents over-forgiving random errors

5. Fairness outcome
- Compares Standard WER vs Lattice-WER per model
- Shows whether a model was unfairly penalized by rigid reference matching

Current result in this run:
- All listed models have delta 0.0000 (Standard WER equals Lattice-WER), indicating no unfairly penalized cases detected in the provided evaluation set under current variant tables.

## Files in This Repository

Core code:
- task_complete.py

Task 1 outputs:
- task1_spelling_classification.csv
- task1_statistics.csv

Task 2 outputs:
- task2_wer_results.csv

Summary:
- RESULTS.md

Inputs:
- uniques_words.txt
- ft_data.xlsx
- question4.xlsx

Dependencies:
- requirements.txt

## How to Run

1. Install dependencies

pip install -r requirements.txt

2. Execute the assignment pipeline

python task_complete.py

This regenerates:
- task1_spelling_classification.csv
- task1_statistics.csv
- task2_wer_results.csv
- RESULTS.md

## Required Files to Run

These files must be present in the same folder before running task_complete.py:
- task_complete.py
- requirements.txt
- uniques_words.txt
- ft_data.xlsx
- question4.xlsx

Generated after run (not required beforehand):
- task1_spelling_classification.csv
- task1_statistics.csv
- task2_wer_results.csv
- RESULTS.md

## Data Access Note

The script follows the provided transcription URL format and processes JSON transcript endpoints accordingly.
