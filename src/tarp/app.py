import datetime
import torch
from pathlib import Path

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import Subset

import plotly.express as px
import polars as pl

from tarp.services.utilities.seed import establish_random_seed
from tarp.cli.logging.colored import ColoredLogger
from tarp.services.tokenizers.pretrained.dnabert import Dnabert2Tokenizer
from tarp.services.datasource.sequence import (
    TabularSequenceSource,
    FastaSliceSource,
)
from tarp.services.datasets.classification.multilabel import (
    MultiLabelClassificationDataset,
)
from tarp.services.datasets.metric.triplet import MultilabelOfflineTripletDataset
from tarp.services.training.classification.multilabel import (
    MultiLabelClassificationTrainer,
)
from tarp.services.evaluation.losses.multilabel import AsymmetricFocalLoss
from tarp.services.training.metric.triplet import TripletMetricTrainer
from tarp.services.preprocessing.augumentation import (
    CombinationTechnique,
    RandomMutation,
    InsertionDeletion,
    ReverseComplement,
)

from tarp.model.finetuning.metric.triplet import TripletMetricModel
from tarp.model.backbone.untrained.hyena import HyenaEncoder
from tarp.model.backbone.untrained.lstm import LstmEncoder
from tarp.model.backbone.pretrained.dnabert2 import FrozenDnabert2Encoder, Dnabert2Encoder
from tarp.model.finetuning.classification import ClassificationModel

from tarp.config import LstmConfig, HyenaConfig, Dnabert2Config


def main() -> None:
    ColoredLogger.info("App started")

    # Set seed for reproducibility
    SEED = establish_random_seed(69420)  # FuNnY NuMbEr :D
    ColoredLogger.info(f"Random seed set to {SEED}")

    label_columns = (
        pl.read_csv(Path("temp/data/processed/labels.csv")).to_series().to_list()
    )

    # Create dataset
    dataset = MultiLabelClassificationDataset(
        (
            TabularSequenceSource(source=Path("temp/data/processed/card_amr.parquet"))
            & FastaSliceSource(
                directory=Path("temp/data/external/sequences"),
                metadata=Path("temp/data/processed/non_amr_genes_10000.parquet"),
                key_column="genomic_nucleotide_accession.version",
                start_column="start_position_on_the_genomic_accession",
                end_column="end_position_on_the_genomic_accession",
                orientation_column="orientation",
            )
        ),
        Dnabert2Tokenizer(),
        sequence_column="sequence",
        label_columns=label_columns,
        maximum_sequence_length=512,
        augumentation=CombinationTechnique(
            [
                RandomMutation(),
                InsertionDeletion(),
                ReverseComplement(0.5),
            ]
        ),
    )

    triplet_dataset = MultilabelOfflineTripletDataset(
        base_dataset=dataset, label_cache=Path("temp/data/interim/labels_cache.parquet")
    )

    # Make a subset of the dataset for quick testing
    train_dataset = Subset(dataset, range(768, len(dataset)))
    valid_dataset = Subset(dataset, range(768))

    # encoder = LstmEncoder(
    #     vocabulary_size=dataset.tokenizer.vocab_size,
    #     embedding_dimension=LstmConfig.embedding_dimension,
    #     hidden_dimension=LstmConfig.hidden_dimension,
    #     padding_id=dataset.tokenizer.pad_token_id,
    #     number_of_layers=LstmConfig.number_of_layers,
    #     dropout=LstmConfig.dropout,
    #     bidirectional=LstmConfig.bidirectional,
    # )

    encoder = Dnabert2Encoder(hidden_dimension=Dnabert2Config.hidden_dimension)

    # encoder = HyenaEncoder(
    #     vocabulary_size=dataset.tokenizer.vocab_size,
    #     embedding_dimension=HyenaConfig.embedding_dimension,
    #     hidden_dimension=HyenaConfig.hidden_dimension,
    #     padding_id=dataset.tokenizer.pad_token_id,
    #     number_of_layers=HyenaConfig.number_of_layers,
    #     dropout=HyenaConfig.dropout,
    # )

    classification_model = ClassificationModel(
        encoder=encoder,
        number_of_classes=len(label_columns),
    )

    triplet_model = TripletMetricModel(
        encoder=encoder,
    )

    triplet_train_dataset = Subset(triplet_dataset, range(768, len(triplet_dataset)))
    triplet_valid_dataset = Subset(triplet_dataset, range(768))

    ColoredLogger.info(
        f"Training {classification_model.encoder.__class__.__name__} model"
    )

    # Use torch compile to optimize the model
    ColoredLogger.info("Compiling model with torch.compile")
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

    ColoredLogger.info("Training model with triplet loss first")
    metric_learning_trainer = TripletMetricTrainer(
        model=triplet_model,
        train_dataset=triplet_train_dataset,
        valid_dataset=triplet_valid_dataset,
        optimizer=optimizer_triplet,
        scheduler=ReduceLROnPlateau(optimizer_triplet, mode="min", patience=3),
        device=device,
        epochs=10,
        num_workers=2,
    )
    metric_learning_trainer.fit()

    classification_model.encoder.load_state_dict(triplet_model.encoder.state_dict())

    ColoredLogger.info("Training model with multi-label classification loss")
    trainer = MultiLabelClassificationTrainer(
        model=classification_model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        optimizer=optimizer_classification,
        scheduler=ReduceLROnPlateau(optimizer_classification, mode="min", patience=3),
        device=device,
        epochs=50,
        num_workers=4,
        batch_size=32,
        # criterion=FocalLoss(alpha=alphas, gamma=2.0),
        criterion=AsymmetricFocalLoss(
            gamma_pos=2,
            gamma_neg=2,  # class_weights=pos_weights.to(device)
        ),
    )
    trainer.fit()

    torch.save(
        classification_model.state_dict(),
        f"temp/checkpoints/{classification_model.encoder.__class__.__name__}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt",
    )

    ColoredLogger.info("Training complete")

    # Visualize training history
    history = trainer.history  # List of dicts
    history_df = pl.DataFrame(history)
    fig = px.line(history_df, y=history_df.columns, title="Training History")
    fig.write_html(
        f"temp/reports/{classification_model.encoder.__class__.__name__}.html"
    )


if __name__ == "__main__":
    main()
