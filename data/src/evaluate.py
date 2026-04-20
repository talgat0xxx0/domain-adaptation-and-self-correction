# -*- coding: utf-8 -*-  #vot1ac23 # klow6667
import torch  #lalafa
import re
import pandas as pd

from datasets import load_from_disk
#from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
from peft import PeftModel

# ======================================================
# PATHS
# ======================================================
VAL_DATA    = "/content/gdrive/MyDrive/trns/gpt_dataset"
#MODEL_PATH  = "/content/gdrive/MyDrive/gemma_baq_pretrained"
#SAVE_CSV    = "/content/gdrive/MyDrive/llama_eval_results_batched.csv"
# ВНИМАНИЕ: Здесь должен быть путь к LoRA, обученной именно для Llama-3.1
#LORA_PATH   = "/content/gdrive/MyDrive/gemma_dapt_sft_v1_25032026/final"


MODEL_PATH  = "/content/gdrive/MyDrive/mt5_saved_cleaned"
SAVE_CSV    = "/content/gdrive/MyDrive/llama_eval_results_batched.csv"
# ВНИМАНИЕ: Здесь должен быть путь к LoRA, обученной именно для Llama-3.1
#LORA_PATH   = "/content/gdrive/MyDrive/gemma_sft_v1_25032026/final2"

BATCH_SIZE  = 6          # 4–8 для 48GB
MAX_LEN     = 2048
MAX_NEW     = 256

# ======================================================
# LOAD DATA
# ======================================================
val_dataset = load_from_disk(VAL_DATA).select(range(200))

# ======================================================
# LOAD TOKENIZER
# ======================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
#tokenizer.pad_token = tokenizer.eos_token

# ======================================================
# LOAD MODEL (4-bit)
# ======================================================

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    #torch_dtype=torch.bfloat16,
    torch_dtype=torch.float16
)
#model = PeftModel.from_pretrained(model, LORA_PATH)
#model.eval()
model.eval()
model.config.use_cache = True

print("🔥 mT5 модель загружена (без LoRA)")

# ======================================================
# UTILS
# ======================================================
def clean_text(text: str) -> str:
    text = re.sub(r"[\n\r]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def clean_mt5_artifacts(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"<extra_id_\d+>", "", text)
    text = re.sub(r"\b(summary|kazakh|revised summary|document)\b\s*[:\-]?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" :,-.\n\t")

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()

def get_ngrams(tokens, n=2):
    return list(zip(*[tokens[i:] for i in range(n)]))

def token_overlap_ratio(doc, summary):
    doc_tokens = set(tokenize(doc))
    sum_tokens = tokenize(summary)
    if len(sum_tokens) == 0:
        return 0
    overlap = sum(1 for t in sum_tokens if t in doc_tokens)
    return overlap / len(sum_tokens)

def bigram_overlap_ratio(doc, summary):
    doc_tokens = tokenize(doc)
    sum_tokens = tokenize(summary)

    doc_bigrams = set(get_ngrams(doc_tokens, 2))
    sum_bigrams = get_ngrams(sum_tokens, 2)

    if len(sum_bigrams) == 0:
        return 0

    overlap = sum(1 for bg in sum_bigrams if bg in doc_bigrams)
    return overlap / len(sum_bigrams)

def ssbuild_verification_prompt(document: str, summary: str) -> str:
    return (
        "check whether the following kazakh summary contains factual errors or hallucinations.\n\n"
        f"document: {document}\n\n"
        f"summary: {summary}\n\n"
        "answer:"
    )

def build_verification_prompt(document: str, summary: str) -> str:
    return (
        "check whether the following kazakh summary contains factual errors or hallucinations.\n\n"
        f"document: {document}\n\n"
        f"summary: {summary}\n\n"
        "answer:"
    )

def build_refinement_prompt(document: str, initial_summary: str) -> str:
    return (
        "correct the kazakh summary. fix factual and grammatical errors. do not add new information.\n\n"
        f"document: {document}\n\n"
        f"summary: {initial_summary}\n\n"
        "corrected summary:"
    )

def ssssssssssssssbuild_prompt(document: str) -> str:
    return (
        "summarize the following news article in kazakh language.\n\n"
        f"{document}\n\n"
        "summary:"
    )

def build_prompt(document: str) -> str:
    return (
        "summarize the following text in kazakh.\n"
        "important: do not copy sentences from the original text. "
        "write an abstractive summary in your own words in 2-3 sentences.\n\n"
        f"text: {document}\n\n"
        "summary:"
    )

# ======================================================
# GENERATION CONFIG
# ======================================================
gen_cfg = dict(
    max_new_tokens=60,
    num_beams=4,
    do_sample=False,
    no_repeat_ngram_size=3
)

gen_cfg2 = dict(
    max_new_tokens=60,
    num_beams=4,
    do_sample=False,
    no_repeat_ngram_size=3
)

# bad words for mt5 sentinel tokens
bad_words_ids = tokenizer(
    [f"<extra_id_{i}>" for i in range(100)],
    add_special_tokens=False
).input_ids

# ======================================================
# BATCHED GENERATION
# ======================================================
preds, refs, docs = [], [], []
print("🚀 Старт batched генерации...")

for start in range(0, len(val_dataset), BATCH_SIZE):
    batch = val_dataset[start:start + BATCH_SIZE]

    docs_batch = [clean_text(x) for x in batch["document"]]
    refs_batch = [clean_text(x) for x in batch["summary"]]

    prompts = [build_prompt(doc) for doc in docs_batch]

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **enc,
            **gen_cfg,
            bad_words_ids=bad_words_ids
        )

    decoded = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )
    decoded = [clean_mt5_artifacts(x) for x in decoded]

    preds.extend(decoded)
    refs.extend(refs_batch)
    docs.extend(docs_batch)

    print(f"✔ {min(start + BATCH_SIZE, len(val_dataset))}/{len(val_dataset)}")

print("✔ Генерация завершена")

# ======================================================
# ВТОРОЙ ПРОХОД: ПРОВЕРКА (Verification)
# ======================================================
verifications = []
print("🔍 Старт проверки генераций на ошибки...")

# ======================================================
# ВТОРОЙ ПРОХОД: УЛУЧШЕНИЕ (Refinement)
# ======================================================
final_preds = []
print("🛠️ Старт улучшения (Refinement)...")

for start in range(0, len(docs), BATCH_SIZE):
    batch_docs = docs[start:start + BATCH_SIZE]
    batch_initial = preds[start:start + BATCH_SIZE]

    r_prompts = [build_refinement_prompt(d, p) for d, p in zip(batch_docs, batch_initial)]

    enc_r = tokenizer(
        r_prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN
    ).to(model.device)

    with torch.no_grad():
        r_outputs = model.generate(
            **enc_r,
            **gen_cfg2,
            bad_words_ids=bad_words_ids
        )

    r_decoded = tokenizer.batch_decode(r_outputs, skip_special_tokens=True)

    #final_preds.extend([clean_text(x) for x in r_decoded])
    # --- ВОТ ЗДЕСЬ ВСТАВЛЯЕМ ОЧИСТКУ ---
    batch_cleaned = []
    for x in r_decoded:
        # 1. Базовая чистка (пробелы, переносы из твоей функции clean_text)
        text = clean_mt5_artifacts(x)

        # 2. Удаляем "вежливые" слова-артефакты в конце предложения
        # Регулярка ищет слова Жақсы, Түзетілді и т.д. только в самом конце строки ($)
        text = re.sub(r"(Жақсы|Түзетілді|Орындалды|Дайын|Рахмет)[\.!\s]*$", "", text, flags=re.IGNORECASE)

        batch_cleaned.append(text.strip())

    # Добавляем уже идеально чистые тексты
    final_preds.extend(batch_cleaned)
    # ----------------------------------
    print(f"✨ Исправлено: {min(start + BATCH_SIZE, len(docs))}/{len(docs)}")

print("✔ Весь процесс завершен")

# ======================================================
# EXTRACTIVENESS ANALYSIS (INITIAL vs FINAL)
# ======================================================

tok_init, bi_init = [], []
tok_final, bi_final = [], []

for d, s_init, s_final in zip(docs, preds, final_preds):
    tok_init.append(token_overlap_ratio(d, s_init))
    bi_init.append(bigram_overlap_ratio(d, s_init))

    tok_final.append(token_overlap_ratio(d, s_final))
    bi_final.append(bigram_overlap_ratio(d, s_final))

print("\n📊 EXTRACTIVENESS COMPARISON")
print(f"Initial → token: {sum(tok_init)/len(tok_init):.3f}, bigram: {sum(bi_init)/len(bi_init):.3f}")
print(f"Final   → token: {sum(tok_final)/len(tok_final):.3f}, bigram: {sum(bi_final)/len(bi_final):.3f}")

# ======================================================
# SAVE CSV
# ======================================================
# ======================================================
# ИТОГОВЫЙ ВЫВОД И СОХРАНЕНИЕ
# ======================================================
print("\n" + "="*30)
print("📊 ПРИМЕРЫ РАБОТЫ И ПРОВЕРКИ:")
print("="*30)

# ======================================================
# ИТОГОВЫЙ ВЫВОД (СРАВНЕНИЕ ДО И ПОСЛЕ)
# ======================================================
print("\n" + "="*50)
print("📊 АНАЛИЗ УЛУЧШЕНИЙ (REFINEMENT):")
print("="*50)

for i in range(min(20, len(docs))):#for i in range(min(5, len(docs))): # Показываем первые 5 для контроля
    print(f"\n🔹 ПРИМЕР {i+1}")
    print(f"✅ Эталон (Reference): {refs[i][:100]}...")
    print(f"1️⃣ Первая попытка (Initial): {preds[i]}")
    print(f"2️⃣ Исправленная версия (Final): {final_preds[i]}")

    # Небольшая проверка: изменился ли текст?
    if preds[i] == final_preds[i]:
        print("🆗 Изменений не потребовалось (текст корректен)")
    else:
        print("✨ Модель внесла правки в мазмұндама")
    print("-" * 30)

# ======================================================
# СОХРАНЕНИЕ
# ======================================================
pd.DataFrame({
    "document": docs,
    "reference": refs,
    "initial_attempt": preds,      # Первая версия
    "final_refined": final_preds   # Исправленная версия
}).to_csv(SAVE_CSV, index=False)

print(f"\n✅ Результаты сохранены в: {SAVE_CSV}")

# ======================================================
# СРАВНИТЕЛЬНЫЙ АНАЛИЗ МЕТРИК (ДО И ПОСЛЕ REFINEMENT)
# ======================================================
print("\n" + "="*50)
print("📊 СРАВНЕНИЕ КАЧЕСТВА: INITIAL vs REFINED")
print("="*50)

# 1. Считаем ROUGE для обоих вариантов
rouge = evaluate.load("rouge")
scores_initial = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
scores_final = rouge.compute(predictions=final_preds, references=refs, use_stemmer=True)

# 2. Считаем BERTScore для обоих вариантов
bert = evaluate.load("bertscore")
# Берем первые 100 (или все 50), чтобы не ждать долго
bs_initial = bert.compute(predictions=preds[:100], references=refs[:100], lang="kk")
bs_final = bert.compute(predictions=final_preds[:100], references=refs[:100], lang="kk")

f1_initial = sum(bs_initial["f1"]) / len(bs_initial["f1"])
f1_final = sum(bs_final["f1"]) / len(bs_final["f1"])

chrf = evaluate.load("chrf")
chrf_initial = chrf.compute(predictions=preds, references=refs)
chrf_final = chrf.compute(predictions=final_preds, references=refs)

# 3. Выводим таблицу сравнения
metrics_data = {
    "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore F1","chrF++"],
    "Initial (Pass 1)": [
        scores_initial["rouge1"],
        scores_initial["rouge2"],
        scores_initial["rougeL"],
        f1_initial,
        chrf_initial["score"]
    ],
    "Refined (Pass 2)": [
        scores_final["rouge1"],
        scores_final["rouge2"],
        scores_final["rougeL"],
        f1_final,
        chrf_final["score"]
    ]
}

df_metrics = pd.DataFrame(metrics_data)

# Считаем разницу в процентах
df_metrics["Improvement (%)"] = ((df_metrics["Refined (Pass 2)"] / df_metrics["Initial (Pass 1)"]) - 1) * 100

print(df_metrics.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))

# ======================================================
# АНАЛИЗ ИЗМЕНЕНИЙ (Сколько текстов было исправлено)
# ======================================================
changed_count = sum(1 for p, f in zip(preds, final_preds) if p != f)
print(f"\n📝 Всего примеров: {len(docs)}")
print(f"✨ Исправлено моделью: {changed_count} ({changed_count/len(docs)*100:.1f}%)")

if changed_count > 0:
    print("\n🔍 Пример первого исправления:")
    for p, f in zip(preds, final_preds):
        if p != f:
            print(f"❌ БЫЛО: {p}")
            print(f"✅ СТАЛО: {f}")
            break