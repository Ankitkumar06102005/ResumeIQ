"""
ner_model.py — Phase 3: Fine-tuned DistilBERT NER for resume entity extraction.

Entities: SKILL, DEGREE, COMPANY, JOB_TITLE

Two components:
  1. RegexNER  — fast rule-based baseline (no GPU needed)
  2. TransformerNER — fine-tuned DistilBERT (needs labeled data + GPU ideally)

To label your own data, use doccano or Label Studio and export in JSONL format,
then convert with the utility at the bottom of this file.
"""

import os
import re
import sys
import warnings
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Entity labels
# ---------------------------------------------------------------------------
LABEL_LIST = ["O", "B-SKILL", "I-SKILL", "B-DEGREE", "I-DEGREE",
              "B-COMPANY", "I-COMPANY", "B-JOB_TITLE", "I-JOB_TITLE"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# ---------------------------------------------------------------------------
# 1. Regex-based NER baseline
# ---------------------------------------------------------------------------

# Degree patterns
_DEGREE_PATTERNS = re.compile(
    r"\b(b\.?s\.?|b\.?a\.?|m\.?s\.?|m\.?a\.?|ph\.?d\.?|m\.?b\.?a\.?|"
    r"bachelor(?:\'s)?|master(?:\'s)?|doctorate|associate(?:\'s)?)\b",
    re.IGNORECASE,
)

# Common job title keywords
_TITLE_PATTERNS = re.compile(
    r"\b(engineer|developer|scientist|analyst|manager|director|architect|"
    r"consultant|specialist|lead|head|intern|associate|senior|junior|staff|"
    r"principal|vp|cto|ceo|coo|ciso)\b",
    re.IGNORECASE,
)

# Tech skills (hard-coded patterns — extend as needed)
_SKILL_PATTERNS = re.compile(
    r"\b(python|java|javascript|typescript|c\+\+|c#|golang|rust|scala|r|"
    r"sql|nosql|postgresql|mysql|mongodb|redis|elasticsearch|"
    r"react|angular|vue|nodejs|django|flask|fastapi|spring|"
    r"aws|gcp|azure|docker|kubernetes|terraform|ansible|jenkins|"
    r"machine learning|deep learning|nlp|computer vision|"
    r"pytorch|tensorflow|scikit-learn|pandas|numpy|spark|kafka|"
    r"git|linux|agile|scrum|rest api|graphql|microservices)\b",
    re.IGNORECASE,
)

# Company indicators (very rough — looks for "Inc", "Ltd", "Corp", etc.)
_COMPANY_PATTERNS = re.compile(
    r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)* (?:Inc\.?|Ltd\.?|Corp\.?|LLC\.?|"
    r"Technologies|Solutions|Systems|Labs|Group|Services|Consulting))\b"
)


class RegexNER:
    """
    Rule-based NER using regular expressions. Fast, no training needed,
    but brittle — misses novel skill names and struggles with context.
    Used as the Phase 3 comparison baseline.
    """

    def extract(self, text: str) -> dict[str, list[str]]:
        return {
            "SKILL": list({m.group().lower() for m in _SKILL_PATTERNS.finditer(text)}),
            "DEGREE": list({m.group().lower() for m in _DEGREE_PATTERNS.finditer(text)}),
            "COMPANY": list({m.group() for m in _COMPANY_PATTERNS.finditer(text)}),
            "JOB_TITLE": list({m.group().lower() for m in _TITLE_PATTERNS.finditer(text)}),
        }

    def evaluate(self, samples: list[dict]) -> dict:
        """
        Compute macro-avg P/R/F1 over a list of labeled samples.
        Each sample: {"text": str, "entities": {"SKILL": [...], ...}}
        """
        from sklearn.metrics import precision_recall_fscore_support

        all_true, all_pred = [], []
        entity_types = ["SKILL", "DEGREE", "COMPANY", "JOB_TITLE"]

        for sample in samples:
            true_ents = sample["entities"]
            pred_ents = self.extract(sample["text"])
            for etype in entity_types:
                true_set = set(true_ents.get(etype, []))
                pred_set = set(pred_ents.get(etype, []))
                all_ents = true_set | pred_set
                for e in all_ents:
                    all_true.append(1 if e in true_set else 0)
                    all_pred.append(1 if e in pred_set else 0)

        if not all_true:
            return {"precision": 0, "recall": 0, "f1": 0}

        p, r, f, _ = precision_recall_fscore_support(
            all_true, all_pred, average="binary", zero_division=0
        )
        return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


# ---------------------------------------------------------------------------
# 2. Transformer NER (fine-tuned DistilBERT)
# ---------------------------------------------------------------------------

class TransformerNER:
    """
    Fine-tuned DistilBERT for token-classification NER.
    Requires labeled data in HuggingFace datasets format with BIO tags.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", model_dir: str = "models/ner"):
        self.model_name = model_name
        self.model_dir = model_dir
        self._model = None
        self._tokenizer = None

    # ---- Training --------------------------------------------------------

    def train(
        self,
        train_dataset,
        eval_dataset,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
    ) -> None:
        """
        Fine-tune DistilBERT on a HuggingFace NER dataset.
        Dataset must have columns: tokens (list[str]), ner_tags (list[int]).
        """
        try:
            from transformers import (
                AutoTokenizer, AutoModelForTokenClassification,
                TrainingArguments, Trainer, DataCollatorForTokenClassification,
            )
            from datasets import DatasetDict
            import torch
        except ImportError as e:
            raise ImportError(f"Phase 3 deps missing: {e}. pip install transformers datasets torch")

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(LABEL_LIST),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )

        def tokenize_and_align(examples):
            tokenized = tokenizer(
                examples["tokens"],
                truncation=True,
                is_split_into_words=True,
                padding="max_length",
                max_length=256,
            )
            labels = []
            for i, label_ids in enumerate(examples["ner_tags"]):
                word_ids = tokenized.word_ids(batch_index=i)
                label_list_aligned = []
                prev_word = None
                for word_id in word_ids:
                    if word_id is None:
                        label_list_aligned.append(-100)  # ignored by loss
                    elif word_id != prev_word:
                        label_list_aligned.append(label_ids[word_id])
                    else:
                        # For sub-tokens, use I- tag if B- was the start
                        tag = label_ids[word_id]
                        tag_name = ID2LABEL[tag]
                        if tag_name.startswith("B-"):
                            tag = LABEL2ID["I-" + tag_name[2:]]
                        label_list_aligned.append(tag)
                    prev_word = word_id
                labels.append(label_list_aligned)
            tokenized["labels"] = labels
            return tokenized

        train_tok = train_dataset.map(tokenize_and_align, batched=True)
        eval_tok = eval_dataset.map(tokenize_and_align, batched=True)

        args = TrainingArguments(
            output_dir=self.model_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            logging_steps=50,
            report_to="none",
        )

        data_collator = DataCollatorForTokenClassification(tokenizer)
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_tok,
            eval_dataset=eval_tok,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
        )

        print("Fine-tuning DistilBERT NER...")
        trainer.train()
        trainer.save_model(self.model_dir)
        tokenizer.save_pretrained(self.model_dir)
        print(f"Model saved → {self.model_dir}")

        self._model = model
        self._tokenizer = tokenizer

    @staticmethod
    def _compute_metrics(p) -> dict:
        """seqeval token-level F1 for NER evaluation."""
        from seqeval.metrics import precision_score, recall_score, f1_score

        predictions, labels = p
        predictions = predictions.argmax(-1)

        true_labels = [
            [ID2LABEL[l] for l in label if l != -100]
            for label in labels
        ]
        true_preds = [
            [ID2LABEL[p_] for p_, l in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]

        return {
            "precision": precision_score(true_labels, true_preds),
            "recall": recall_score(true_labels, true_preds),
            "f1": f1_score(true_labels, true_preds),
        }

    # ---- Inference -------------------------------------------------------

    def load(self) -> None:
        """Load a fine-tuned model from disk."""
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self._model = AutoModelForTokenClassification.from_pretrained(self.model_dir)
        self._model.eval()

    def extract(self, text: str) -> dict[str, list[str]]:
        """Run inference and return extracted entities grouped by type."""
        if self._model is None or self._tokenizer is None:
            self.load()

        import torch
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self._model(**inputs)

        preds = outputs.logits.argmax(-1)[0].tolist()
        tokens = self._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        entities: dict[str, list[str]] = {k: [] for k in ["SKILL", "DEGREE", "COMPANY", "JOB_TITLE"]}
        current_entity = []
        current_label = None

        for token, pred_id in zip(tokens, preds):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue

            label = ID2LABEL[pred_id]

            if label.startswith("B-"):
                if current_entity and current_label:
                    _flush_entity(entities, current_entity, current_label)
                current_entity = [token.replace("##", "")]
                current_label = label[2:]
            elif label.startswith("I-") and current_label:
                current_entity.append(token.replace("##", ""))
            else:
                if current_entity and current_label:
                    _flush_entity(entities, current_entity, current_label)
                current_entity = []
                current_label = None

        return entities


def _flush_entity(entities: dict, tokens: list, label: str) -> None:
    text = " ".join(tokens).strip()
    if text and label in entities:
        entities[label].append(text)


# ---------------------------------------------------------------------------
# Utility: convert doccano/Label Studio export → HuggingFace dataset
# ---------------------------------------------------------------------------

def build_hf_dataset_from_jsonl(jsonl_path: str):
    """
    Convert a JSONL file (from doccano) with {"text": ..., "label": [[start, end, type]]}
    into a HuggingFace dataset with token-level BIO tags.
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("pip install datasets")

    import json

    records = []
    with open(jsonl_path) as f:
        for line in f:
            item = json.loads(line)
            text = item["text"]
            spans = item.get("label", [])
            tokens, tags = _span_to_bio(text, spans)
            records.append({"tokens": tokens, "ner_tags": [LABEL2ID[t] for t in tags]})

    return Dataset.from_list(records)


def _span_to_bio(text: str, spans: list) -> tuple[list[str], list[str]]:
    """Convert character-level spans to word-level BIO tags."""
    words = text.split()
    char_to_word = {}
    pos = 0
    for i, word in enumerate(words):
        for j in range(len(word)):
            char_to_word[pos + j] = i
        pos += len(word) + 1  # +1 for space

    tags = ["O"] * len(words)
    for start, end, label in spans:
        entity_type = label.upper().replace(" ", "_")
        word_indices = sorted(set(
            char_to_word[c] for c in range(start, end)
            if c in char_to_word
        ))
        for k, wi in enumerate(word_indices):
            prefix = "B-" if k == 0 else "I-"
            full_tag = prefix + entity_type
            if full_tag in LABEL2ID:
                tags[wi] = full_tag

    return words, tags
