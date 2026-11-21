import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from packages.model import CodeTransformer, ModelConfig


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig.load(ROOT / "configs/model/cpu-test.json")

    def test_shape_and_causal_forward(self):
        model = CodeTransformer(self.config)
        ids = torch.randint(0, self.config.vocab_size, (2, 17))
        logits = model(ids)
        self.assertEqual(logits.shape, (2, 17, self.config.vocab_size))
        self.assertTrue(torch.isfinite(logits).all())

    def test_save_load_equivalence(self):
        model = CodeTransformer(self.config).eval()
        ids = torch.randint(0, self.config.vocab_size, (1, 9))
        with torch.no_grad():
            expected = model(ids)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(model.state_dict(), path)
            restored = CodeTransformer(self.config).eval()
            restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            with torch.no_grad():
                actual = restored(ids)
        self.assertTrue(torch.equal(expected, actual))

    def test_parameter_count_and_tied_embeddings(self):
        model = CodeTransformer(self.config)
        self.assertGreater(model.num_parameters(), 0)
        self.assertIs(model.token_embeddings.weight, model.lm_head.weight)

    def test_context_limit(self):
        model = CodeTransformer(self.config)
        ids = torch.zeros((1, self.config.max_position_embeddings + 1), dtype=torch.long)
        with self.assertRaises(ValueError):
            model(ids)


if __name__ == "__main__":
    unittest.main()
