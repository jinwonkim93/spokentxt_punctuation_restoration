import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

punctuation_map = {
    "LABEL_0": "",
    "LABEL_1": ",",
    "LABEL_2": ".",
    "LABEL_3": "?",
    "LABEL_4": "!",
}


class PunctuationModel:
    def __init__(
        self, model_name: str = "whooray/koen_punctuation", device: str = "cpu"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.device = device
        self.model = self.model.to(device)

    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits  # shape: (1, seq_len, num_labels)

        pred_ids = torch.argmax(logits, dim=2).squeeze().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(
            inputs["input_ids"].squeeze().tolist()
        )
        word_groups = []
        current_word = ""
        current_punct = ""

        for i, token in enumerate(tokens):
            if token in self.tokenizer.all_special_tokens:
                continue

            if i == 0 or token.startswith("▁"):
                if current_word:
                    if current_word and current_word[-1] == current_punct:
                        word_groups.append(current_word)
                    else:
                        word_groups.append(current_word + current_punct)
                current_word = ""
                current_punct = ""
                token = token[1:]

            current_word += token
            predicted_label = (
                self.model.config.id2label[pred_ids[i]]
                if hasattr(self.model.config, "id2label")
                else f"LABEL_{pred_ids[i]}"
            )
            current_punct = punctuation_map.get(predicted_label, "")

        if current_word:
            if current_word and current_word[-1] == current_punct:
                word_groups.append(current_word)
            else:
                word_groups.append(current_word + current_punct)
        return " ".join(word_groups)

    def __call__(self, text: str):
        return self.predict(text)
