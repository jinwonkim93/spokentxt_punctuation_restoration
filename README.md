# Spoken text Punctuation Restoration

# Languages
English, Korean

# Model details
https://huggingface.co/whooray/koen_punctuation

# Install
```bash
pip install spokentxt-punctuation-restoration
```

# Usage
```python
from spokentxt_punctuation_restoration import PunctuationModel

model = PunctuationModel(model_name = "whooray/koen_punctuation", device = "cpu") # device = cuda:0
model("안녕하세요")
#'안녕하세요.'
model("Hello how are you")
#'Hello, how are you?'
```