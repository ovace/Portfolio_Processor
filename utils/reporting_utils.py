# utils/reporting_utils.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Iterable, Optional, Set
import json
import logging

logger = logging.getLogger("portfolio_utils")

@dataclass
class FileReport:
    input_path: str
    structure: str  # "standard" | "hybrid" | "unknown"
    tables_detected: int = 0
    total_rows_out: int = 0
    # Per-table quick peek (first 2 rows as dicts) – useful in DEBUG
    table_samples: List[Dict[str, List[Dict[str, object]]]] = field(default_factory=list)

    # Column introspection
    original_columns: Set[str] = field(default_factory=set)
    normalized_columns: Set[str] = field(default_factory=set)
    output_fields: List[str] = field(default_factory=list)

    # Mappings & deltas
    rename_map: Dict[str, str] = field(default_factory=dict)  # original -> canonical
    missing_canonical: List[str] = field(default_factory=list)
    unused_input_columns: List[str] = field(default_factory=list)

    # Counters for diagnostics
    rows_read_raw: int = 0
    rows_after_cleanup: int = 0
    rows_removed_blank: int = 0
    rows_removed_duplicates: int = 0

    # Optional notes
    notes: List[str] = field(default_factory=list)

    def add_table_sample(self, idx: int, df):
        try:
            sample = df.head(2).fillna("").to_dict(orient="records")
            self.table_samples.append({f"table_{idx}_sample": sample})
        except Exception:
            pass

    def finalize_columns(
        self,
        output_fields: Iterable[str],
        col_map_used: Dict[str, str],  # original -> canonical applied
    ):
        self.output_fields = list(output_fields)
        self.rename_map = dict(col_map_used)

        canonical_set = set(self.output_fields)
        self.missing_canonical = [c for c in self.output_fields if c not in self.normalized_columns]

        # Columns present in input but not used in normalized set (excludes those renamed to canonical)
        mapped_originals = set(col_map_used.keys())
        self.unused_input_columns = sorted(
            list((self.original_columns - mapped_originals) - self.normalized_columns)
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "input_path": self.input_path,
            "structure": self.structure,
            "tables_detected": self.tables_detected,
            "rows_read_raw": self.rows_read_raw,
            "rows_after_cleanup": self.rows_after_cleanup,
            "rows_removed_blank": self.rows_removed_blank,
            "rows_removed_duplicates": self.rows_removed_duplicates,
            "total_rows_out": self.total_rows_out,
            "original_columns": sorted(list(self.original_columns)),
            "normalized_columns": sorted(list(self.normalized_columns)),
            "output_fields": self.output_fields,
            "rename_map": self.rename_map,
            "missing_canonical": self.missing_canonical,
            "unused_input_columns": self.unused_input_columns,
            "notes": self.notes,
            "table_samples": self.table_samples,
        }

    def log_summary(self, level: int = logging.INFO):
        payload = json.dumps(self.to_dict(), indent=2)
        logger.log(level, f"AUDIT: File summary\n{payload}")


@dataclass
class OverallReport:
    files: List[FileReport] = field(default_factory=list)

    def add(self, fr: FileReport):
        self.files.append(fr)

    def aggregate(self) -> Dict[str, object]:
        total_rows = sum(f.total_rows_out for f in self.files)
        total_after_cleanup = sum(f.rows_after_cleanup for f in self.files)
        total_removed_blank = sum(f.rows_removed_blank for f in self.files)
        total_removed_dupes = sum(f.rows_removed_duplicates for f in self.files)
        total_tables = sum(f.tables_detected for f in self.files)

        # Union of originals/normalized, and a frequency map for missing canonical fields
        originals_union = sorted(list(set().union(*[f.original_columns for f in self.files if f.original_columns])))
        normalized_union = sorted(list(set().union(*[f.normalized_columns for f in self.files if f.normalized_columns])))
        missing_freq: Dict[str, int] = {}
        for f in self.files:
            for c in f.missing_canonical:
                missing_freq[c] = missing_freq.get(c, 0) + 1

        return {
            "files_processed": len(self.files),
            "tables_detected_total": total_tables,
            "rows_out_total": total_rows,
            "rows_after_cleanup_total": total_after_cleanup,
            "rows_removed_blank_total": total_removed_blank,
            "rows_removed_duplicates_total": total_removed_dupes,
            "original_columns_union": originals_union,
            "normalized_columns_union": normalized_union,
            "missing_canonical_frequency": missing_freq,
        }

    def log_overall(self, level: int = logging.INFO):
        payload = json.dumps(self.aggregate(), indent=2)
        logger.log(level, f"AUDIT: Overall summary\n{payload}")
