# Response Clarity Classification — SemEval 2026 Task 6

Classifying how clearly a politician answers a journalist's question. Three labels:
`Clear Reply`, `Ambivalent`, `Clear Non-Reply`. Dataset:
[ailsntua/QEvasion](https://huggingface.co/datasets/ailsntua/QEvasion). Task page:
[CLARITY — SemEval 2026 Task 6](https://sites.google.com/view/clarity-semeval2026/).

The repo covers four approaches I ran end to end on the same split:

1. TF-IDF and averaged Word2Vec with Logistic Regression — the sanity-check baselines.
2. Fine-tuned sentence-pair transformers (BERT, DistilBERT, DeBERTa-v3), written as a
   plain PyTorch loop rather than through `Trainer`.
3. Prompting Qwen2.5-3B and Qwen2.5-7B with zero-shot, chain-of-thought and few-shot
   templates.
4. A D3-agentic pipeline on Qwen2.5-0.5B-Instruct: four specialised agents passing
   structured JSON to each other, with a deterministic decision rule at the end.

The final Kaggle submission is a 6-encoder ensemble with one round of pseudo-labelling
on top of the stage-2 code. It landed **2nd on the private leaderboard**.

## Results

Macro F1 is the task metric. "Validation" = my own stratified split from `train`
(seed 42); "Kaggle" = the official held-out test.

| Approach                              | Val macro F1 | Kaggle macro F1 |
|---------------------------------------|-------------:|----------------:|
| TF-IDF + LogReg                       |            — |            0.55 |
| Word2Vec (avg) + LogReg               |            — |            0.55 |
| DistilBERT fine-tuned                 |        0.584 |            0.62 |
| DeBERTa-v3-base fine-tuned            |        0.455 |            0.59 |
| BERT-base fine-tuned                  |    **0.623** |        **0.67** |
| Qwen2.5-7B few-shot                   |        0.460 |               — |
| D3-agentic (Qwen2.5-0.5B)             |  0.30 – 0.42 |               — |
| Encoder ensemble + pseudo-labels      |            — | **0.67** (2nd)  |

Per-class F1 for the best models:

| Method                      | Clear Reply | Ambivalent | Clear Non-Reply |
|-----------------------------|------------:|-----------:|----------------:|
| BERT-base-uncased           |       0.604 |      0.696 |           0.569 |
| DistilBERT-base-uncased     |       0.525 |      0.662 |           0.565 |
| DeBERTa-v3-base             |       0.299 |      0.632 |           0.435 |
| Qwen2.5-7B / few-shot       |       0.342 |      0.560 |           0.478 |

All training curves, confusion matrices and the EDA plots are in `results/`. The raw
numbers I pulled from the project reports are in `results/metrics.csv`.

## What I learned

- **BERT-base is still the right default** for a 3.4k-sample sentence-pair task. The
  larger DeBERTa-v3-base collapsed toward the Ambivalent class (predicted ~69% of the
  Clear Reply samples as Ambivalent) and lost about 17 F1 points on validation. The
  usual story — too few samples for the capacity.
- **Clear Reply vs Ambivalent is the hard boundary.** Every model's biggest confusion
  lives here. Clear Non-Reply is actually recovered reasonably well, mostly thanks to
  the weighted cross-entropy.
- **Chain-of-thought hurt me on this task.** Both Qwen2.5-3B and 7B *lost* F1 going
  from zero-shot to CoT — the model over-interprets any hedging as evasion and starts
  under-predicting Clear Reply. Few-shot was the best prompting setup by a clear
  margin.
- **Compounding error killed D3-agentic at this scale.** Four JSON hops at ~85%
  reliability per agent compose to roughly 52% joint. The decomposition idea is clean
  on paper but a 0.5B backbone cannot carry four chained agents. With a 7B backbone I
  suspect this would look very different.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py                                   # fetches QEvasion from HF
python scripts/run_baselines.py --method tfidf --variant bigrams
python scripts/run_baselines.py --method w2v
python scripts/run_finetune.py  --model bert                      # or distilbert / deberta
python scripts/run_prompting.py --model 7b --strategy few-shot --use-4bit
python scripts/run_d3_agentic.py --save-logs
python scripts/run_ensemble.py  --with-pseudo                     # the Kaggle submission
```

All seeds are fixed to 42 (`src/clarity/data.py::set_seed`). Fine-tuning needs
~12 GB VRAM for BERT/DistilBERT and ~16 GB for DeBERTa or the 7B prompting runs
(`--use-4bit` cuts that down). The ensemble script really only makes sense on a
T4/A100-class GPU — the DeBERTa-v3-large folds are the bottleneck.

## Repo layout

```
src/clarity/     data, evaluate, baselines, finetune, prompting, agents, ensemble
scripts/         one CLI per experiment, all with --help
notebooks/       cleaned EDA + error-analysis notebooks
results/         metrics.csv + confusion matrices, training curves, EDA plots
```

## Limitations and things I'd do differently

The dataset is small — around 3.4k training pairs for a 3-way imbalanced task.
Everything on top of BERT-base is fighting that ceiling, the ensemble included.
Given more time I'd focus on data augmentation and on using conversation history
(previous turns) rather than throwing larger backbones at the problem.

Two caveats on the numbers above. The D3-agentic row is given as a range rather than
a single value because the result moves a lot between the ablation and prompt-variant
runs and I never settled on one headline configuration. And the ensemble/pseudo
Kaggle score is a different system from D3-agentic — the two share only
`src/clarity/data.py`; they end up in the same repo because they were both parts of
the final homework.

## Dataset license

QEvasion is distributed on Hugging Face under the terms listed on its
[dataset card](https://huggingface.co/datasets/ailsntua/QEvasion). This repo does not
redistribute any part of it — `scripts/download_data.py` pulls it from HF at run time.
Review the dataset licence before downstream use.
