import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from packages.tokenizer import ByteBPETokenizer


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ByteBPETokenizer()
        self.tokenizer.train([
            "def greet(name: str) -> str:\n    return f'Hello {name}'\n",
            "export interface Item { id: number; name: string }\n",
            "function add(a: number, b: number) {\n  return a + b;\n}\n",
        ], vocab_size=384)

    def test_special_token_ids_are_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            self.tokenizer.save(directory)
            restored = ByteBPETokenizer.load(directory)
        for token in self.tokenizer.special_tokens:
            self.assertEqual(
                restored.special_token_id(token),
                self.tokenizer.special_token_id(token),
            )

    def test_round_trip_code_features(self):
        text = "def café():\n\tif True:\n        return {'a': [1, 2]}\n"
        decoded = self.tokenizer.decode(self.tokenizer.encode(text))
        self.assertEqual(decoded, text)

    def test_fim_special_tokens(self):
        text = "prefix <|fim_prefix|> middle <|fim_suffix|> suffix <|fim_middle|>"
        ids = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(ids)
        self.assertIn("<|fim_prefix|>", decoded)
        self.assertIn("<|fim_suffix|>", decoded)
        self.assertIn("<|fim_middle|>", decoded)


if __name__ == "__main__":
    unittest.main()
