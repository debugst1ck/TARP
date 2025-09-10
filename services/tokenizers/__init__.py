from torch import Tensor

class Tokenizer:
    def tokenize(self, text: str) -> Tensor:
        """
        Tokenizes the input text.

        :param str text: The text to tokenize.
        :return Tensor: A tensor containing the tokenized input.
        """
        raise NotImplementedError
    
    @property
    def pad_token_id(self) -> int:
        raise NotImplementedError
    
    def train(self, texts: list[str], vocab_size: int = 32000, **kwargs):
        """
        Optional: Train tokenizer on provided texts.
        Only implemented for trainable subclasses.
        """
        raise NotImplementedError