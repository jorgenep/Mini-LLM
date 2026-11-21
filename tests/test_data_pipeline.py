import sys
import unittest

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from packages.data.pipeline import SourceRecord, deduplicate_records, stable_split, prepare_record
from packages.tokenizer import ByteBPETokenizer


def record(text: str, path: str = "src/example.py") -> SourceRecord:
    return SourceRecord(
        source_id="test", repository="test/repo", path=path,
        license="MIT", license_reference="https://example.invalid/license",
        collection_date="2026-11-21", language="python", text=text,
        filtering_decisions=[], transformation_history=[],
    )


class DataPipelineTests(unittest.TestCase):
    def test_exact_deduplication(self):
        records = [record("def x():\n    return 1\n"), record("def x():\n    return 1\n")]
        self.assertEqual(len(deduplicate_records(records)), 1)

    def test_stable_split_is_reproducible(self):
        records = [record(f"def f{i}():\n    return {i}\n") for i in range(20)]
        first = stable_split(records, 1729, (0.8, 0.1, 0.1))
        second = stable_split(records, 1729, (0.8, 0.1, 0.1))
        fingerprints = lambda split: [r.fingerprint for r in split]
        self.assertEqual([fingerprints(s) for s in first], [fingerprints(s) for s in second])

    def test_psm_and_spm_fim(self):
        tokenizer = ByteBPETokenizer()
        source = record("def add(a, b):\n    return a + b\n")
        psm = prepare_record(source, tokenizer, "psm_fim")
        spm = prepare_record(source, tokenizer, "spm_fim")
        self.assertEqual(psm["objective"], "psm_fim")
        self.assertEqual(spm["objective"], "spm_fim")
        self.assertIn(tokenizer.special_token_id("<|fim_middle|>"), psm["input_ids"])


if __name__ == "__main__":
    unittest.main()
