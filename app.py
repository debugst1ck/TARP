import torch
from torch import nn, Tensor

from pathlib import Path

from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from services.tokenizers.pretrained.dnabert import Dnabert2Tokenizer
from services.datasource.sequence import TabularSequenceDataSource
from services.datasets.finetuning.classification import MultiLabelClassificationDataset
from services.training.finetuning.multilabel.dnabert import FinetuningClassificationTrainer

from services.loggers.colored import ColoredLogger

from safetensors.torch import save_model, load_model

import polars as pl

from model.finetuning.classification.dnabert2 import Dnabert2ClassificationModel
from model.finetuning.classification.lstm import LstmClassificationModel

from services.preprocessing.augumentation import CombinationTechnique, RandomMutation, IndelAugmentation, ReverseComplement

from torch.utils.data import Subset

import plotly.express as px

def app() -> None:
    ColoredLogger.info("App started")
    
    # Set seed for reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(SEED)
    ColoredLogger.info(f"Set random seed to {SEED}")

    df = pl.read_parquet("temp/data/preprocessed/card_cleaned.parquet")
    # Get every column except 'sequence' as label columns
    label_columns = [col for col in df.collect_schema().names() if col != "sequence"]
    
    class_weights = []
    N = df.height
    for col in label_columns:
        n_i = df[col].sum()
        w_i = (N - n_i) / (n_i + 1e-5)  # pos_weight for BCE
        class_weights.append(w_i)
        
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    class_weights /= class_weights.max()  # Normalize to [0, 1]
            
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
        )
    )
    
    # Make a subset of the dataset for quick testing
    valid_dataset = Subset(dataset, range(256))
    train_dataset = Subset(dataset, range(256, len(dataset)))

    model = LstmClassificationModel(
        vocabulary_size=dataset.tokenizer.vocab_size,
        number_of_classes=len(label_columns),
        embedding_dimension=256,
        hidden_dimension=128,
        padding_id=dataset.tokenizer.pad_token_id,
        number_of_layers=3,
        dropout=0.2,
        bidirectional=True,
    )
    
    torch.compile(model)
    
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)
    
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        ColoredLogger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        ColoredLogger.warning("Using CPU")
        device = torch.device("cpu")
    
    trainer = FinetuningClassificationTrainer(
        model = model,
        train_dataset = train_dataset,
        valid_dataset = valid_dataset,
        optimizer = optimizer,
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3),
        device=device,
        epochs=10,
        num_workers=2,
        batch_size=16,
        class_weights=class_weights,
    )
    
    trainer.fit()
    
    # Save the model as safetensors
    # Check if directory exists
    Path("temp/models").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "temp/models/lstm_classifier.pth")
    
    ColoredLogger.info("Training complete")

    # Visualize training history
    history = trainer.history # List of dicts
    history_df = pl.DataFrame(history)
    fig = px.line(history_df, y=history_df.columns, title="Training History")
    fig.write_html("temp/models/lstm_training_history.html")
    


if __name__ == "__main__":
    app()
