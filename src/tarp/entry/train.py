#! /usr/bin/env python3
from tarp.services.datasets.classification.multilabel import (
    MultiLabelClassificationDataset,
)
from tarp.services.datasource.sequence import (
    TabularSequenceSource,
    CombinationSource,
    FastaSliceSource,
)
from tarp.services.preprocessing.augmentation import (
    CombinationTechnique,
    RandomMutation,
    InsertionDeletion,
    ReverseComplement,
)

from tarp.services.tokenizers.pretrained.dnabert import Dnabert2Tokenizer
from tarp.services.preprocessing.augmentation import (
    CombinationTechnique,
    RandomMutation,
    InsertionDeletion,
    ReverseComplement,
)

import polars as pl
from pathlib import Path


def main():
    label_columns = (
        pl.read_csv(Path("temp/data/processed/labels.csv")).to_series().to_list()
    )

    dataset = MultiLabelClassificationDataset(
        CombinationSource(
            [
                TabularSequenceSource(
                    source=Path("temp/data/processed/card_amr.parquet")
                ),
                FastaSliceSource(
                    directory=Path("temp/data/external/sequences"),
                    metadata=Path(
                        "temp/data/processed/protein_coding_genes_5000_test_with_non_amr.parquet"
                    ),
                    key_column="genomic_nucleotide_accession.version",
                    start_column="start_position_on_the_genomic_accession",
                    end_column="end_position_on_the_genomic_accession",
                    orientation_column="orientation",
                ),
            ]
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
    
    pass


if __name__ == "__main__":
    main()
