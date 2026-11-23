"""Real BERTScore API smoke with a tiny local random model, not quality validation."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

from evaluation.feedback import similarity

AVAILABLE = all(importlib.util.find_spec(n) is not None for n in ('torch', 'transformers', 'bert_score', 'rouge_score', 'sacrebleu'))


@unittest.skipUnless(AVAILABLE, 'Install requirements-evaluation-bert.txt')
class BertScoreIntegrationTests(unittest.TestCase):
    def test_local_model_runs_and_refuses_silent_long_text_truncation(self):
        import torch
        from transformers import BertConfig, BertModel, BertTokenizer
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            words = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'clear', 'feedback', 'add', 'an', 'example', '.']
            (path / 'vocab.txt').write_text('\n'.join(words))
            tokenizer = BertTokenizer(vocab_file=str(path / 'vocab.txt'), model_max_length=16)
            tokenizer.save_pretrained(path)
            torch.manual_seed(42)
            model = BertModel(BertConfig(vocab_size=len(words), hidden_size=16, num_hidden_layers=1,
                                        num_attention_heads=2, intermediate_size=32, max_position_embeddings=16))
            model.save_pretrained(path)
            text = 'clear feedback add an example .'
            rows = [{'id': 'fixture', 'valid': True, 'prediction': text, 'reference': text}]
            result = similarity(rows, bert_model=str(path), bert_layers=1)
            self.assertAlmostEqual(result['bertscore']['f1'], 1, places=5)
            self.assertIn('f1', rows[0]['bertscore'])
            with self.assertRaisesRegex(ValueError, 'no silent truncation'):
                similarity([{'id': 'too-long', 'valid': True, 'prediction': 'feedback ' * 40, 'reference': text}],
                           bert_model=str(path), bert_layers=1)
