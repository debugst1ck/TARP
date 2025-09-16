import torch
from torch import nn, Tensor

from pathlib import Path

from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from services.tokenizers.pretrained.dnabert import Dnabert2Tokenizer
from services.datasource.sequence import TabularSequenceDataSource
from services.datasets.finetuning.classification import MultiLabelClassificationDataset
from services.training.classification.multilabel import MultiLabelClassificationTrainer

from services.loggers.colored import ColoredLogger

from services.evaluation.losses.multilabel import AsymmetricFocalLoss

import polars as pl

from model.finetuning.classification.dnabert2 import FrozenDnabert2ClassificationModel
from model.finetuning.classification.lstm import LstmClassificationModel

from services.preprocessing.augumentation import (
    CombinationTechnique,
    RandomMutation,
    IndelAugmentation,
    ReverseComplement,
)

from torch.utils.data import Subset

import plotly.express as px

import random
import numpy as np


def app() -> None:
    ColoredLogger.info("App started")

    # Set seed for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(SEED)
    ColoredLogger.info(f"Set random seed to {SEED}")

    df = pl.read_parquet("temp/data/preprocessed/card_cleaned.parquet")
    # Get every column except 'sequence' as label columns
    label_columns = [col for col in df.collect_schema().names() if col != "sequence"]

    N = df.height
    alphas, pos_weights = [], []

    for col in label_columns:
        n_i = df[col].sum()  # number of positives
        frac_pos = n_i / N
        frac_neg = 1 - frac_pos

        # ---- For FocalLoss (alpha balancing factor) ----
        # Higher alpha when positives are rare
        alpha_i = frac_neg / (frac_pos + frac_neg)
        alphas.append(alpha_i)

        # ---- For BCEWithLogitsLoss (pos_weight) ----
        # Weight positive examples higher when rare
        pos_w = frac_neg / (frac_pos + 1e-8)
        pos_weights.append(pos_w)

    alphas = torch.tensor(alphas, dtype=torch.float)
    pos_weights = torch.tensor(pos_weights, dtype=torch.float)

    # Delete df to save memory
    del df

    # Create dataset
    dataset = MultiLabelClassificationDataset(
        TabularSequenceDataSource(
            source=Path("temp/data/preprocessed/card_cleaned.parquet")
        ),
        Dnabert2Tokenizer(),
        sequence_column="sequence",
        label_columns=label_columns,
        maximum_sequence_length=768,
        augumentation=CombinationTechnique(
            [
                RandomMutation(),
                IndelAugmentation(),
                ReverseComplement(0.5),
            ]
        ),
    )

    # Make a subset of the dataset for quick testing
    valid_dataset = Subset(dataset, range(768))
    train_dataset = Subset(dataset, range(768, len(dataset)))

    model = LstmClassificationModel(
        vocabulary_size=dataset.tokenizer.vocab_size,
        number_of_classes=len(label_columns),
        embedding_dimension=768,
        hidden_dimension=512,
        padding_id=dataset.tokenizer.pad_token_id,
        number_of_layers=3,
        dropout=0.2,
        bidirectional=True,
    )

    # model = FrozenDnabert2ClassificationModel(
    #     number_of_classes=len(label_columns),
    #     hidden_dimension=768,
    # )

    torch.compile(model)

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        ColoredLogger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        ColoredLogger.warning("Using CPU")
        device = torch.device("cpu")

    trainer = MultiLabelClassificationTrainer(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        optimizer=optimizer,
        scheduler=ReduceLROnPlateau(optimizer, mode="min", patience=3),
        device=device,
        epochs=10,
        num_workers=4,
        batch_size=32,
        # criterion=FocalLoss(alpha=alphas, gamma=2.0),
        criterion=AsymmetricFocalLoss(gamma_pos=0, gamma_neg=2, class_weights=pos_weights)
    )

    trainer.fit()

    # Save the model as safetensors
    # Check if directory exists
    Path("temp/models").mkdir(parents=True, exist_ok=True)
    Path("temp/history").mkdir(parents=True, exist_ok=True)
    
    torch.save(
        model.state_dict(), f"temp/models/{model.__class__.__name__}.pt"
    )

    ColoredLogger.info("Training complete")

    # Visualize training history
    history = trainer.history  # List of dicts
    history_df = pl.DataFrame(history)
    fig = px.line(history_df, y=history_df.columns, title="Training History")
    fig.write_html(f"temp/history/{model.__class__.__name__}.html")

if __name__ == "__main__":
    app()
