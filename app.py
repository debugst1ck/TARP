import torch
from pathlib import Path

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from services.tokenizers.pretrained.dnabert import Dnabert2Tokenizer
from services.datasource.sequence import TabularSequenceDataSource
from services.datasets.classification.multilabel import MultiLabelClassificationDataset
from services.datasets.metric.triplet import MultiLabelTripletDataset
from services.training.classification.multilabel import MultiLabelClassificationTrainer

from services.training.metric.triplet import TripletMetricTrainer
from model.finetuning.metric.triplet import TripletMetricModel

from services.loggers.colored import ColoredLogger

from services.evaluation.losses.multilabel import AsymmetricFocalLoss

import polars as pl

from model.backbone.untrained.lstm import LstmEncoder

from model.backbone.pretrained.dnabert2 import Dnabert2Encoder, FrozenDnabert2Encoder
from model.finetuning.classification import ClassificationModel

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
        maximum_sequence_length=512,
        augumentation=CombinationTechnique(
            [
                RandomMutation(),
                IndelAugmentation(),
                ReverseComplement(0.5),
            ]
        ),
    )

    triplet_dataset = MultiLabelTripletDataset(base_dataset=dataset)

    # Make a subset of the dataset for quick testing
    valid_dataset = Subset(dataset, range(768))
    train_dataset = Subset(dataset, range(768, len(dataset)))

    encoder = LstmEncoder(
        vocabulary_size=dataset.tokenizer.vocab_size,
        embedding_dimension=768,
        hidden_dimension=512,
        padding_id=dataset.tokenizer.pad_token_id,
        number_of_layers=3,
        dropout=0.2,
        bidirectional=True,
    )

    classification_model = ClassificationModel(
        encoder=encoder,
        number_of_classes=len(label_columns),
    )

    triplet_model = TripletMetricModel(
        encoder=encoder,
    )

    triplet_train_dataset = Subset(triplet_dataset, range(768, len(triplet_dataset)))
    triplet_valid_dataset = Subset(triplet_dataset, range(768))

    # model = ClassificationModel(
    #     FrozenDnabert2Encoder(hidden_dimension=768),
    #     number_of_classes=len(label_columns),
    # )

    ColoredLogger.info(
        f"Training {classification_model.encoder.__class__.__name__} model"
    )

    # Use torch compile to optimize the model
    torch.compile(classification_model, mode="reduce-overhead")
    torch.compile(triplet_model, mode="reduce-overhead")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        ColoredLogger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        ColoredLogger.warning("Using CPU for training, this may be slow")
        device = torch.device("cpu")

    optimizer_classification = AdamW(
        classification_model.parameters(), lr=0.001, weight_decay=1e-2
    )
    optimizer_triplet = AdamW(triplet_model.parameters(), lr=0.001, weight_decay=1e-2)

    metric_learning_trainer = TripletMetricTrainer(
        model=triplet_model,
        train_dataset=triplet_train_dataset,
        valid_dataset=triplet_valid_dataset,
        optimizer=optimizer_triplet,
        scheduler=ReduceLROnPlateau(optimizer_triplet, mode="min", patience=3),
        device=device,
        epochs=1,
        num_workers=2,
    )
    metric_learning_trainer.fit()

    classification_model.encoder.load_state_dict(triplet_model.encoder.state_dict())

    trainer = MultiLabelClassificationTrainer(
        model=classification_model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        optimizer=optimizer_classification,
        scheduler=ReduceLROnPlateau(optimizer_classification, mode="min", patience=3),
        device=device,
        epochs=9,
        num_workers=4,
        batch_size=32,
        # criterion=FocalLoss(alpha=alphas, gamma=2.0),
        criterion=AsymmetricFocalLoss(
            gamma_pos=2, gamma_neg=2, class_weights=pos_weights.to(device)
        ),
    )

    trainer.fit()

    # Save the model as safetensors
    # Check if directory exists
    Path("temp/models").mkdir(parents=True, exist_ok=True)
    Path("temp/history").mkdir(parents=True, exist_ok=True)

    torch.save(
        classification_model.state_dict(),
        f"temp/models/{classification_model.encoder.__class__.__name__}.pt",
    )

    ColoredLogger.info("Training complete")

    # Visualize training history
    history = trainer.history  # List of dicts
    history_df = pl.DataFrame(history)
    fig = px.line(history_df, y=history_df.columns, title="Training History")
    fig.write_html(f"temp/history/{classification_model.encoder.__class__.__name__}.html")


if __name__ == "__main__":
    app()
