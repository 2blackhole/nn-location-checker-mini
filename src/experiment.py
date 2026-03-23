"""Experiment data model and CSV handler for parsing and serializing log output.

This module provides the `Experiment` dataclass, which parses structured
log lines produced by the training pipeline and exposes results as a
CSV-serializable object, and `ExperimentCSVHandler`, a context manager
for appending experiments to a CSV file.

Intended usage::

    experiment = Experiment()
    with subprocess.Popen(
        ...,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    ) as training:
        for line in training.stderr:
            ...
            experiment.update(line)

    with ExperimentCSVHandler("results.csv") as handler:
        handler.writerow(experiment)
"""

import csv
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import TracebackType
from typing import IO, ClassVar, Self

from dataset import Marker

__all__ = ["Experiment", "ExperimentCSVHandler"]


@dataclass
class Experiment:
    """A single experiment parsed from log output.

    Parses structured log lines into typed fields and supports
    direct serialization to CSV via iteration.

    Attributes:
        donor: Identifier of the model.
        segment: Donor's segment in ``start:end`` format.
        classifier: Configuration of the classifier used.
        accuracy: Classification accuracy as a string-encoded float.
        avg_time_per_image: Mean classification time per image in seconds.
        macro_f1: Macro-averaged F1 score across all classes.
        macro_f1_per_class: Per-class F1 scores as a list of strings.

    Example:
        >>> exp = Experiment()
        >>> exp.update("17/03/2024 14:26:57 INFO:Donor: AlexNet")
        >>> exp.donor
        'AlexNet'
        >>> dict(exp)
        {'donor': 'AlexNet', 'segment': '', ...}
    """

    donor: str = ""
    segment: str = ""
    classifier: str = ""
    accuracy: str = ""
    avg_time_per_image: str = ""
    macro_f1: str = ""
    macro_f1_per_class: list[str] = field(default_factory=list)

    _FIELD_PATTERNS: ClassVar[dict[str, re.Pattern[str]]] = {
        "donor": re.compile(r"Donor: (?P<data>\w*)"),
        "segment": re.compile(r"Segment: (?P<data>\d*:\d*)"),
        "classifier": re.compile(r"Classifier: (?P<data>\[.*?\])"),
        "accuracy": re.compile(r"Accuracy: (?P<data>\d*\.\d*)"),
        "macro_f1_per_class": re.compile(r"Macro f1 per class: (?P<data>\[.*\])"),
        "macro_f1": re.compile(r"Macro f1: (?P<data>\d*.\d*)"),
        "avg_time_per_image": re.compile(r"Average time per image: (?P<data>\d*.\d*)"),
    }

    def update(self, line: str) -> None:
        """Parse a log line and update the matching field.

        Extracts the message segment after the third colon and matches it
        against known patterns, updating the corresponding field in place.

        Args:
            line: A raw log line in the format ``DD/MM/YYYY HH:MM:SS level:message``.

        Example:
            >>> exp = Experiment()
            >>> exp.update("17/03/2024 14:26:57 INFO:Donor: AlexNet")
            >>> exp.donor
            'AlexNet'
        """
        message = line.split(":", 3)[-1].strip()

        for field_name, pattern in self._FIELD_PATTERNS.items():
            if matched := pattern.match(message):
                data = matched.group("data")
                if field_name == "macro_f1_per_class":
                    setattr(self, field_name, data[1:-1].split(", "))
                else:
                    setattr(self, field_name, data)
                return

    @classmethod
    def header(cls) -> list[str]:
        """Return the CSV column names in serialization order.

        Returns:
            A list of field name strings, matching the order that
            ``__iter__`` yields values.

        Example:
            >>> Experiment.headers()
            ['donor', 'segment', 'classifier', ...]
        """
        base = [
            f.name
            for f in fields(cls)
            if not f.name.startswith("_") and f.name != "macro_f1_per_class"
        ]
        f1_per_class = [f"macro_f1_class_{i}" for i in range(len(Marker))]

        return base + f1_per_class

    def __iter__(self) -> Iterator[tuple[str, str]]:
        """Iterate over ``(field_name, value)`` pairs for CSV serialization.

        Elements of list fields (e.g. ``macro_f1_per_class``)
        are serialized individually.

        Yields:
            Tuple[str, str]: A ``(field_name, value)`` pair for each
            public field.

        Example:
            >>> dict(Experiment(donor="ABC", accuracy="0.95"))
            {'donor': 'ABC', 'accuracy': '0.95', ...}
        """
        for f in fields(self):
            if f.name.startswith("_"):
                continue

            value = getattr(self, f.name)

            if f.name.endswith("per_class"):
                for i, score in zip(range(len(Marker)), value, strict=True):
                    yield f"macro_f1_class_{i}", score
            else:
                yield f.name, value


class ExperimentCSVHandler:
    """Context manager for appending experiments to a CSV file.

    Creates the file with a header on first use, appends rows on
    subsequent runs — analogous to ``logging.FileHandler``.

    Example:
        >>> exp = Experiment()
        >>> ...
        >>> with ExperimentCSVHandler("results.csv") as handler:
        ...     handler.write(exp)
    """

    def __init__(self, path: Path) -> None:
        """Initialize a handler.

        Args:
            path: Path to the target CSV file. Created with a header row if it
            does not exist; rows are appended on subsequent runs.
        """
        self._path = path
        self._file: None | IO = None
        self._writer: None | csv.DictWriter[str] = None

    def __enter__(self) -> Self:
        """Open the CSV file and return the handler for use."""
        file_exists = self._path.exists()
        self._file = self._path.open("a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=Experiment.header())

        if not file_exists:
            self._writer.writeheader()

        return self

    def writerow(self, experiment: Experiment) -> None:
        """Write a single experiment row to the CSV."""
        if self._writer is None:
            raise RuntimeError("Run this method with context manager")
        self._writer.writerow(dict(experiment))

    def writerows(self, experiments: Iterable[Experiment]) -> None:
        """Write multiple experiment rows to the CSV."""
        if self._writer is None:
            raise RuntimeError("Run this method with context manager")
        self._writer.writerows(dict(exp) for exp in experiments)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the CSV file, flushing any remaining buffered writes."""
        if self._file:
            self._file.close()
