import torch
from torch import nn, Tensor

from pathlib import Path

from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from services.tokenizers.pretrained.dnabert import Dnabert2Tokenizer
from model.finetuning.classification.dnabert2 import SimpleClassifier
from services.datasource.sequence import TabularSequenceDataSource
from services.datasets.finetuning.classification import MultiLabelClassificationDataset
from services.training.finetuning.multilabel.dnabert import Dnabert2FinetuningClassificationTrainer

from services.loggers.colored import ColoredLogger

from safetensors.torch import save_model, load_model

import polars as pl


def app():
    ColoredLogger.info("Starting training")
    df = pl.scan_parquet("temp/data/preprocessed/card_cleaned.parquet")
    # Get every column except 'sequence' as label columns
    label_columns = [col for col in df.collect_schema().names() if col != "sequence"]
    
    dataset = MultiLabelClassificationDataset(
        TabularSequenceDataSource(
            source=Path("temp/data/preprocessed/card_cleaned.parquet")
        ),
        Dnabert2Tokenizer(),
        sequence_column="sequence",
        label_columns=label_columns,
        maximum_sequence_length=512,
    )
    
    model = SimpleClassifier(
        number_of_classes=len(label_columns),
        embedding_dimension=768,
    )
    torch.compile(model)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # For each item in the dataset, print the sequence and labels
    for title, tensor in dataset.__getitem__(0).items():
        ColoredLogger.debug(f"{title}: {tensor.size()}")
    
    trainer = Dnabert2FinetuningClassificationTrainer(
        model = model,
        train_dataset = dataset,
        valid_dataset = dataset,
        optimizer = optimizer,
        scheduler = None,
        device=device,
        epochs = 1,
        num_workers=0,
    )
    
    trainer.fit()
    
    # Save the model as safetensors
    save_model(model.state_dict(), "temp/models/dnabert2_multilabel_classification.safetensors",)

    
if __name__ == "__main__":
    app()
