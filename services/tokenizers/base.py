from transformers import AutoTokenizer
from torch import Tensor


class Tokenizer:
    def tokenize(self, text: str) -> Tensor:
        """
        Tokenizes the input text.

        :param str text: The text to tokenize.
        :return Tensor: A tensor containing the tokenized input.
        """
        return NotImplementedError
    
    @property
    def pad_token_id(self) -> int:
        return NotImplementedError


class Dnabert2Tokenizer(Tokenizer):
    """
    Wrapper around the DNABERT-2 tokenizer.
    """
    def __init__(self, name: str = "zhihan1996/DNABERT-2-117M"):
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

    def tokenize(self, text: str) -> Tensor:
        return self.tokenizer(text, return_tensors="pt")["input_ids"][0]

    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id