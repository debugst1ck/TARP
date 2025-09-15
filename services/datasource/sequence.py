import polars as pl
from pathlib import Path
from typing import Optional
from Bio import SeqIO


class SequenceDataSource:
    """
    Encapsulate file interactions for sequence datasets.
    """

    @property
    def height(self) -> int:
        """
        Get the number of rows in the data source.

        :return: The height of the sequence.
        :rtype: int
        """
        raise NotImplementedError

    def retrieve(self, index: int) -> dict:
        """
        Retrieve a single row from the data source.

        :param int index: The index of the row to retrieve.
        :return: A dictionary representation of the row.
        :rtype: dict
        """
        raise NotImplementedError


class TabularSequenceDataSource(SequenceDataSource):
    """
    Reads from a tabular data source (e.g., CSV, Excel, Parquet). Stores in a Polars DataFrame.
    """

    def __init__(self, source: Path):
        self.source = source
        self.dataframe: Optional[pl.DataFrame] = None
        self._read_source()

    def _read_source(self) -> None:
        if self.source.suffix == ".csv":
            self.dataframe = pl.read_csv(self.source)
        elif self.source.suffix == ".xlsx":
            self.dataframe = pl.read_excel(self.source)
        elif self.source.suffix == ".parquet":
            self.dataframe = pl.read_parquet(self.source)
        else:
            raise ValueError(f"Unsupported file type: {self.source.suffix}")

    @property
    def height(self) -> int:
        if self.dataframe is not None:
            return self.dataframe.height
        return 0

    def retrieve(self, index: int) -> dict:
        if self.dataframe is not None:
            return self.dataframe.row(index, named=True)
        return {}


class FastaDirectoryDataSource(SequenceDataSource):
    """
    Reads from a directory of FASTA files. Metadata is stored as a Tabular source.
    """

    def __init__(
        self,
        directory: Path,
        metadata: Path,
        key_column: str,
        sequence_column: str = "sequence",
    ):
        self.directory = directory
        self.metadata = metadata
        self.key_column = key_column
        self.sequence_column = sequence_column

        self.dataframe: Optional[pl.DataFrame] = None
        self._read_source()

    def _read_source(self) -> None:
        if self.metadata.suffix == ".csv":
            self.dataframe = pl.read_csv(self.metadata)
        elif self.metadata.suffix == ".xlsx":
            self.dataframe = pl.read_excel(self.metadata)
        elif self.metadata.suffix == ".parquet":
            self.dataframe = pl.read_parquet(self.metadata)
        else:
            raise ValueError(f"Unsupported file type: {self.metadata.suffix}")

    @property
    def height(self) -> int:
        if self.dataframe is not None:
            return self.dataframe.height
        return 0

    def retrieve(self, index: int) -> dict:
        # Retrieve the key column which correspond to Fasta file name
        if self.dataframe is None:
            return {}

        key = self.dataframe[index][self.key_column]
        fasta_path = self.directory / f"{key}.fasta"

        if not fasta_path.exists():
            return {}

        with open(fasta_path, "r") as handle:
            row = self.dataframe[index].to_dict(as_series=False)
            row[self.sequence_column] = "".join(
                str(record.seq) for record in SeqIO.parse(handle, "fasta")
            )
        return row


class CombinationDataSource(SequenceDataSource):
    """
    Combines multiple data sources into one.
    """

    def __init__(self, sources: list[SequenceDataSource]):
        self.sources = sources

    @property
    def height(self) -> int:
        return sum(source.height for source in self.sources)

    def retrieve(self, index: int) -> dict:
        for source in self.sources:
            if index < source.height:
                return source.retrieve(index)
            index -= source.height
        raise IndexError("Index out of range for combined data sources.")


class InMemorySequenceDataSource(SequenceDataSource):
    """
    Stores sequences in memory for fast access.
    """

    def __init__(self, data: pl.DataFrame):
        self.dataframe = data

    @property
    def height(self) -> int:
        return self.dataframe.height

    def retrieve(self, index: int) -> dict:
        return self.dataframe.row(index, named=True)
