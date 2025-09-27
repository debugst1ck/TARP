# Untrained tokenizer implementation
from services.tokenizers import Tokenizer
from typing import Sequence


class UntrainedTokenizer(Tokenizer):
    def train(self, texts: Sequence[str], vocab_size: int = 32000, **kwargs) -> None:
        """
        Optional: Train tokenizer on provided texts.
        Only implemented for trainable subclasses.

        :param Sequence[str] texts: List of texts to train the tokenizer on.
        :param int vocab_size: Desired vocabulary size.
        :param kwargs: Additional parameters for training.
        :return: None
        """
        raise NotImplementedError
