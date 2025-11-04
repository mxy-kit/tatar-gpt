# TatarGPT: Tiny LLM Trained from Scratch


This repository demonstrates training a compact decoder‑only GPT model **from scratch** on a low‑resource language (Tatar). The project covers:

* custom corpus cleaning & statistics;
* tokenizer design and comparison;
* full training loop ;
* efficiency experiments (bf16, gradient checkpointing);
* decoding & mini‑benchmark.

---

## 📋 Table of Contents

* [1. Corpus Preparation](#-1-corpus-preparation)
* [2. Tokenizer Training](#-2-tokenizer-training-byte-level-bpe)
* [3. Model Configuration & Training](#-3-model-configuration--training)
* [4. Optimization Experiment](#-4-optimization-experiment)
* [5. Evaluation & Generation](#-5-model-evaluation--generation)
* [6. Experiments](#-6-experiments)
* [7. Mini‑Benchmark](#-7-mini-benchmark)


---

##  1. Corpus Preparation

* **Language:** `Tatar (tt)`
* **Source:** [Leipzig Corpora Collection](https://wortschatz-leipzig.de/en/download) — `tat_mixed_2015_1M.tar.gz`
* **Corpus size after cleaning:** ~100 MB
* **Number of lines:** 999,197
* **Average length:** 104 characters

**Cleaning pipeline**

```python
basic_clean()       # HTML unescape, normalize spacing/punctuation
too_numeric()       # remove overly numeric lines (>60% digits)
likely_tatarish()   # keep lines with >=55% Cyrillic and unique Tatar letters
deduplication       # remove duplicates by lowercase + normalized whitespace
```


**Corpus statistics**

| Metric           | Value       |
| ---------------- | ----------- |
| Total lines      | 999,197     |
| Total characters | 104,168,737 |
| Avg. length      | 104.25      |
| Median length    | 95.0        |

---

## 2. Tokenizer Training (Byte‑Level BPE)

Tokenizer trained from scratch using `tokenizers` (Byte‑Level BPE).

**Settings**

* **Model:** BPE
* **Special tokens:** `<pad>`, `<bos>`, `<eos>`, `<unk>`
* **Minimum frequency:** 2
* **Vocab sizes tested:** 8k / 16k / 32k

**Tokenizer comparison**

| Vocab size | Avg tokens / sentence | Low‑freq ratio | Conclusion                     |
| ---------: | --------------------: | -------------: | ------------------------------ |
|         8k |                 26.31 |          0.025 | too fragmented                 |
|        16k |                 23.46 |         0.0088 | ✅ best trade‑off               |
|        32k |                 21.68 |         0.0028 | slightly better, heavier model |

→ **Chosen tokenizer:** `16k` vocabulary — `bpe_16000/tokenizer.json`

---

## 3. Model Configuration & Training

**Model type:** GPT2‑like decoder‑only (trained from scratch on cleaned Tatar corpus).

| Parameter               | Value                |
| ----------------------- | -------------------- |
| Layers × Heads × Hidden | 8 × 8 × 512          |
| Context window          | 512                  |
| Vocabulary              | 16k                  |
| Optimizer               | AdamW                |
| Learning rate           | 3e‑4                 |
| Epochs                  | 1                    |
| Precision               | bf16                 |
| Gradient checkpointing  | Enabled            |
| Training framework      | Hugging Face Trainer |

**Training loss dynamics**

| Step | Train Loss | Val Loss |
| ---: | ---------: | -------: |
|  200 |       6.27 |     6.13 |
|  400 |       5.58 |     5.51 |
|  800 |       5.03 |     4.99 |
| 1200 |       4.72 |     4.68 |
| 1600 |       4.49 |     4.48 |
| 2000 |       4.35 |     4.34 |
| 2400 |       4.24 |     4.24 |
| 2800 |       4.19 |     4.19 |

**Validation Perplexity:** `exp(4.19) ≈ 66.0`
✅ Stable convergence and consistent improvement.

---

## 4. Optimization Experiment

| Setting                        | GPU Memory | Speed (steps/s) | Validation PPL |
| ------------------------------ | ---------: | --------------: | -------------: |
| baseline (fp32, no checkpoint) |    ~7.8 GB |              24 |           66.3 |
| bf16 + gradient checkpointing  |    ~5.2 GB |              26 |           66.0 |

✅ bf16 + checkpointing reduce memory and slightly improve speed without harming quality.
### Model and Tokenizer

My custom **tokenizer** and **model weights** have been uploaded to Hugging Face for reproducibility:  
- Optimal Model weights and tokenizer: [https://huggingface.co/xinyuema/tt-gpt-small2](https://huggingface.co/xinyuema/tt-gpt-small2)  
- Base model weights and tokenizer: [https://huggingface.co/xinyuema/tt-gpt-small](https://huggingface.co/xinyuema/tt-gpt-small)

---

## 💬 5. Model Evaluation & Generation

**Evaluate perplexity**

```python
from math import exp

eval_res = trainer.evaluate(eval_dataset=lm_val)
print("Perplexity:", exp(eval_res["eval_loss"]))
```

**Generation helper** (beam search / top‑p sampling, digit filtering, repetition & length penalties):

```python
outputs = generate(
    prompt,
    max_new_tokens=80,
    method="beam|topp",
    top_p=0.9,
    temperature=0.9,
    no_repeat_ngram_size=3,
    bad_words_ids=digit_ids,
    repetition_penalty=1.1,
    length_penalty=1.0,
)
```

**Examples**

| Prompt (Tatar)                               | Model Answer (sampled)                                                              |
| -------------------------------------------- | ----------------------------------------------------------------------------------- |
| «Казан шәһәре мэры кем?»                     | «Бу хакта ‘Татар-информ’ хәбәрчесенә Казан шәһәре мэры **Илсур Метшин** хәбәр итә.» |
| «ТР Дәүләт Советы Рәисе турында кыскача яз.» | «ТР Президенты **Минтимер Шәймиев** һәм **Рөстәм Миңнеханов** катнашты.»            |

✅ The model produces fluent, grammatically sound Tatar sentences.

---

## 6. Experiments

1. **Tokenizer selection** — 16k gives the best compactness/efficiency balance.
2. **Training optimization** — bf16 + checkpointing ≈30% lower memory, same loss.
3. **Decoding strategies**

| Method         | Style             | Notes           |
| -------------- | ----------------- | --------------- |
| Beam search    | factual, focused  | concise answers |
| Top‑p sampling | diverse, creative | richer text     |

4. **Ablation: `max_new_tokens`**

| Tokens | Behavior                 |
| -----: | ------------------------ |
|     40 | short and precise        |
|     80 | balanced                 |
|    160 | long, slightly off‑topic |

---

## 🧩 7. Mini‑Benchmark

| Metric  | Meaning                | Score |
| ------- | ---------------------- | ----: |
| notEcho | no question repetition |   0.0 |
| kwCover | keyword coverage       |   1.0 |

✅ Model identifies key facts (e.g., «Илсур Метшин», «Казан»).

---

## 🚀 8. Model Upload








