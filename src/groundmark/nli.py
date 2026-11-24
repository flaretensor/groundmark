from pathlib import Path

NLI_MODELS = {
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli": "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli": "6f5cf0a2b59cabb106aca4c287eed12e357e90eb",
}
NLI_LABELS = ("supports", "refutes", "not_enough_info")
NLI_CONFIG_LABELS = ("entailment", "contradiction", "neutral")


class NLIModel:
    def __init__(self, model_dir: Path, *, device: str, dtype: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_dir, local_files_only=True, dtype=getattr(torch, dtype)
            )
            .to(device)
            .eval()
        )
        names = {name.casefold(): index for index, name in self.model.config.id2label.items()}
        if set(names) != set(NLI_CONFIG_LABELS):
            raise ValueError(f"Unrecognized NLI labels: {sorted(names)}")
        self.order = [names[label] for label in NLI_CONFIG_LABELS]
        self.max_length = min(512, self.model.config.max_position_embeddings)

    def score(self, passages: list[str], claims: list[str]) -> tuple[list[list[float]], list[int]]:
        lengths = self.tokenizer(
            passages, claims, truncation=False, return_length=True, verbose=False
        )["length"]
        inputs = self.tokenizer(
            passages,
            claims,
            padding=True,
            truncation="only_first",
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with self.torch.inference_mode():
            probabilities = self.model(**inputs).logits.float().softmax(dim=-1)
        return probabilities[:, self.order].tolist(), lengths
