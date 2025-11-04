# TatarGPT: Tiny LLM Trained from Scratch


This repository demonstrates training a compact decoder‑only GPT model **from scratch** on a low‑resource language (Tatar). The project covers:

* custom corpus cleaning & statistics;
* tokenizer design and comparison;
* full training loop ;
* efficiency experiments (bf16, gradient checkpointing);
* decoding & mini‑benchmark.

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

**Validation Perplexity:** 

After training, the model was evaluated on the validation set using `Trainer.evaluate()`:


```

| Metric           | Value  | Interpretation                                   |
|------------------|--------|--------------------------------------------------|
| Validation loss  | 4.19   | Steadily decreased throughout training           |
| Perplexity       | ≈66.0  | Coherent token prediction; learned basic syntax  |

✅ The loss curve shows smooth convergence without signs of overfitting.  
✅ Perplexity is reasonable for a small from-scratch model trained on ~100 MB text.


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

## 5. Model Generation

### 5.1 Text Generation Setup


Generation function (beam search + repetition penalty + digit suppression):

```python
import re

def generate(prompt, max_new_tokens=80):
    enc = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = mdl.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            bad_words_ids=digit_ids or None,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

---

### 5.2 Example Generations

🟢 **Single factual prompt**

| Prompt (Tatar)                               | Model Answer (sampled)                                                              |
| -------------------------------------------- | ----------------------------------------------------------------------------------- |
| «Казан шәһәре мэры кем?»                     | «Бу хакта ‘Татар-информ’ хәбәрчесенә Казан шәһәре мэры **Илсур Метшин** хәбәр итә.» |
| «ТР Дәүләт Советы Рәисе турында кыскача яз.» | «ТР Президенты **Минтимер Шәймиев** һәм **Рөстәм Миңнеханов** катнашты.»            |

✅ The model produces fluent, grammatically sound Tatar sentences.

🟢 **Paraphrased mini-benchmark**

| Prompt                                   | Model Response (excerpt)           |
|------------------------------------------|------------------------------------|
| Казан шәһәре мэры кем?                   | … Казан шәһәре мэры Илсур Метшин … |
| Казан шәһәре мэры турында кыскача яз.    | … Казан шәһәре мэры Илсур Метшин … |
| Казан шәһәре мэры — ул кем?              | … Казан шәһәре мэры Илсур Метшин … |
| Казан шәһәре мэры турында өч җөмлә яз.   | … Казан шәһәре мэры Илсур Метшин … |

✅ Across multiple phrasings, outputs remain factual and grammatical, indicating the model captures semantic meaning rather than only memorizing surface forms. The model gives the same answer regardless of how the question is phrased.

---

### 5.3 Ablation: max_new_tokens

Different `max_new_tokens` were tested: **40 / 80 / 160**

- 40 → shorter, more focused outputs  
- 80–160 → news-style paragraphs; longer sentences with prompt repetition or topic drift  
- All variants underperformed the default configuration in factual precision and relevance

✅ **Conclusion:** The model can sustain longer, coherent text (good contextual continuity), but default parameters yield the most accurate and relevant answers.

---

### 5.4 Decoding Comparison: Beam Search vs Top-p Sampling

| Strategy     | Expected Characteristics         | Actual Observation                                                                 |
|--------------|----------------------------------|-------------------------------------------------------------------------------------|
| Beam Search  | Deterministic; syntactically stable | Tends to repeat the prompt; generic/incomplete sentences lacking key facts         |
| Top-p        | More diverse and creative        | News-like, verbose text; irrelevant entities; semantic drift (e.g., “... Ульяновски …”) |

✅ **Conclusion:** Both decoding strategies performed worse than the default setup:

- **Beam Search** → short but uninformative, prompt-echoing  
- **Top-p** → longer yet off-topic  

➡️ Therefore, retaining the **default decoding parameters** is most reliable for this task.









