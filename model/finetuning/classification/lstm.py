from torch import nn, Tensor
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        number_of_classes: int,
    ):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, number_of_classes)

    def forward(self, sequence: Tensor, attention_mask: Tensor = None) -> Tensor:
        # sequence shape: (batch_size, seq_length, input_size)
        # Attention mask is used to ignore padding token
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu() # (batch_size,)
            packed_sequence = pack_padded_sequence(
                sequence, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, (hn, cn) = self.lstm(packed_sequence)
            output, _ = pad_packed_sequence(packed_output, batch_first=True)
        else:
            output, (hn, cn) = self.lstm(sequence)
            
        # Use the last hidden state for classification
        logits = self.fc(hn[-1])
        return logits
