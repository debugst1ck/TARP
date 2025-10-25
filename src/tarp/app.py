import datetime
import os
import torch
from pathlib import Path

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

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
from tarp.services.datasets.metric.triplet import MultiLabelOfflineTripletDataset
from tarp.services.training.trainer.classification.multilabel import (
    MultiLabelClassificationTrainer,
)
from tarp.services.evaluation.losses.multilabel import AsymmetricFocalLoss
from tarp.services.training.trainer.metric.triplet import TripletMetricTrainer
from tarp.services.preprocessing.augmentation import (
    CombinationTechnique,
    RandomMutation,
    InsertionDeletion,
    ReverseComplement,
)

from tarp.model.finetuning.metric.triplet import TripletMetricModel
from tarp.model.backbone.untrained.hyena import HyenaEncoder
from tarp.model.backbone.untrained.lstm import LstmEncoder
from tarp.model.backbone.pretrained.dnabert2 import (
    FrozenDnabert2Encoder,
    Dnabert2Encoder,
)
from tarp.model.finetuning.classification import ClassificationModel

from tarp.config import LstmConfig, HyenaConfig, Dnabert2Config

from tarp.services.training.trainer.multitask.triplet_multilabel import (
    JointTripletClassificationTrainer,
)

from sklearn.model_selection import train_test_split

import torch.multiprocessing as mp

def main() -> None:
    ColoredLogger.info("App started")
    
    try:
        mp.set_start_method("spawn", force=True)
        ColoredLogger.info("Multiprocessing start method set to 'spawn'")
        persistent_workers = False
    except RuntimeError:
        ColoredLogger.warning("Multiprocessing start method was already set, skipping...")
        persistent_workers = True
        pass

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

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
            + FastaSliceSource(
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
        augmentation=CombinationTechnique(
            [
                RandomMutation(),
                InsertionDeletion(),
                ReverseComplement(0.5),
            ]
        ),
    )

    dataset_amr_non_amr = MultiLabelClassificationDataset(
        (
            TabularSequenceSource(
                source=Path("temp/data/processed/card_amr_binary.parquet")
            )
            + FastaSliceSource(
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
        label_columns=["amr", "non_amr"],
        maximum_sequence_length=512,
        augmentation=CombinationTechnique(
            [
                RandomMutation(),
                InsertionDeletion(),
                ReverseComplement(0.5),
            ]
        ),
    )

    triplet_dataset_amr_non_amr = MultiLabelOfflineTripletDataset(
        base_dataset=dataset_amr_non_amr,
        label_cache=Path("temp/data/cache/labels_cache_amr_non_amr.parquet"),
    )

    triplet_dataset = MultiLabelOfflineTripletDataset(
        base_dataset=dataset, label_cache=Path("temp/data/cache/labels_cache.parquet")
    )

    label_cache = pl.read_parquet(Path("temp/data/cache/labels_cache.parquet"))
    label_tensor = torch.tensor(label_cache.select(label_columns).to_numpy())

    class_counts = label_tensor.sum(dim=0)
    total_counts = label_tensor.size(0)
    pos_weights = (total_counts - class_counts) / class_counts
    pos_weights = (pos_weights - pos_weights.min()) / (
        pos_weights.max() - pos_weights.min()
    )
    pos_weights = pos_weights * (3.0 - 0.1) + 0.1  # Scale to [0.1, 3.0]

    # Display pos weights as a polars DataFrame
    ColoredLogger.debug(
        str(
            pl.DataFrame(
                {
                    "label": label_columns,
                    "pos_weight": pos_weights.tolist(),
                }
            )
        )
    )

    # Make a subset of the dataset for quick testing
    indices = list(range(len(dataset)))
    train_indices, temp_indices = train_test_split(
        indices, test_size=0.2, random_state=SEED
    )
    valid_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, random_state=SEED
    )
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)
    # test_dataset = Subset(dataset, test_indices)

    # encoder = LstmEncoder(
    #     vocabulary_size=dataset.tokenizer.vocab_size,
    #     embedding_dimension=LstmConfig.embedding_dimension,
    #     hidden_dimension=LstmConfig.hidden_dimension,
    #     padding_id=dataset.tokenizer.pad_token_id,
    #     number_of_layers=LstmConfig.number_of_layers,
    #     dropout=LstmConfig.dropout,
    #     bidirectional=LstmConfig.bidirectional,
    # )

    # encoder = FrozenDnabert2Encoder(hidden_dimension=Dnabert2Config.hidden_dimension)

    encoder = HyenaEncoder(
        vocabulary_size=dataset.tokenizer.vocab_size,
        embedding_dimension=HyenaConfig.embedding_dimension,
        hidden_dimension=HyenaConfig.hidden_dimension,
        padding_id=dataset.tokenizer.pad_token_id,
        number_of_layers=HyenaConfig.number_of_layers,
        dropout=HyenaConfig.dropout,
    )

    classification_model = ClassificationModel(
        encoder=encoder,
        number_of_classes=len(label_columns),
    )

    triplet_model = TripletMetricModel(
        encoder=encoder,
    )

    triplet_train_dataset = Subset(triplet_dataset, train_indices)
    triplet_valid_dataset = Subset(triplet_dataset, valid_indices)

    # triplet_test_dataset = Subset(triplet_dataset, test_indices)

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

    encoder_params = []
    head_params = []
    for name, p in classification_model.named_parameters():
        if "encoder" in name:
            encoder_params.append(p)
        else:
            head_params.append(p)

    bin_triplet_optimizer = AdamW(triplet_model.parameters(), lr=1e-4, weight_decay=1e-2)
    ColoredLogger.info("Starting triplet model training on AMR vs non-AMR task")
    TripletMetricTrainer(
        model=triplet_model,
        train_dataset=Subset(triplet_dataset_amr_non_amr, train_indices),
        valid_dataset=Subset(triplet_dataset_amr_non_amr, valid_indices),
        optimizer=bin_triplet_optimizer,
        scheduler=CosineAnnealingWarmRestarts(bin_triplet_optimizer, T_0=5, T_mult=2),
        device=device,
        epochs=5,
        num_workers=4,
        batch_size=64,
        accumulation_steps=4,
        persistent_workers=persistent_workers,
    ).fit()

    ColoredLogger.info("Starting triplet model training on full multi-label task")
    multi_triplet_optimizer = AdamW(triplet_model.parameters(), lr=1e-4, weight_decay=1e-3)
    TripletMetricTrainer(
        model=triplet_model,
        train_dataset=triplet_train_dataset,
        valid_dataset=triplet_valid_dataset,
        optimizer=multi_triplet_optimizer,
        scheduler=CosineAnnealingWarmRestarts(multi_triplet_optimizer, T_0=5, T_mult=2),
        device=device,
        epochs=15,
        num_workers=4,
        batch_size=64,
        accumulation_steps=4,
        persistent_workers=persistent_workers,
    ).fit()

    # Load the trained weights into the classification model's encoder
    classification_model.encoder.load_state_dict(triplet_model.encoder.state_dict())
    ColoredLogger.info("Starting classification model training")
    optimizer_classification = AdamW(
        [
            {"params": encoder_params, "lr": 1e-4, "weight_decay": 1e-2},
            {"params": head_params, "lr": 1e-3, "weight_decay": 1e-2},
        ]
    )
    trainer = MultiLabelClassificationTrainer(
        model=classification_model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        optimizer=optimizer_classification,
        scheduler=ReduceLROnPlateau(optimizer_classification, mode="min", patience=3),
        criterion=AsymmetricFocalLoss(gamma_neg=1, gamma_pos=3, class_weights=pos_weights),
        device=device,
        epochs=15,
        num_workers=4,
        batch_size=64,
        accumulation_steps=4,
        persistent_workers=persistent_workers,
    )
    trainer.fit()


    torch.save(
        classification_model.state_dict(),
        f"temp/checkpoints/{classification_model.encoder.__class__.__name__}_{run_id}.pt",
    )

    ColoredLogger.info("Training complete")

    # Visualize training history
    history = trainer.context.state.history  # List of dicts
    history_df = pl.DataFrame(history)
    fig = px.line(history_df, y=history_df.columns, title="Training History")
    fig.write_html(
        f"temp/reports/{classification_model.encoder.__class__.__name__}_{run_id}.html"
    )


if __name__ == "__main__":
    main()
