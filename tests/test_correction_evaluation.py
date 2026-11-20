import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.correction import compare_m2, f_scores, m2_sources, run_m2, run_parallel
from evaluation.text_io import TASKS

HAS_ERRANT = importlib.util.find_spec('errant') is not None
HAS_SPACY_MODEL = importlib.util.find_spec('en_core_web_sm') is not None
REF = 'S He go home .\nA 1 2|||R:VERB:SVA|||goes|||REQUIRED|||-NONE-|||0\n\n'
NOOP = 'S He go home .\nA -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\n\n'
WRONG = 'S He go home .\nA 1 2|||R:VERB:SVA|||went|||REQUIRED|||-NONE-|||0\n\n'


class CorrectionBasicTests(unittest.TestCase):
    def test_precision_recall_and_f_half_hand_calculation(self):
        result = f_scores(2, 1, 2)
        self.assertAlmostEqual(result['precision'], 2 / 3)
        self.assertEqual(result['recall'], 0.5)
        self.assertAlmostEqual(result['f0_5'], 0.625)
        self.assertEqual(f_scores(0, 0, 0)['f0_5'], 1)

    def test_m2_empty_malformed_and_out_of_range_edits_rejected(self):
        for text in ('', 'S He goes .\n\n', REF.replace('1 2', '1 99'), REF.replace('|||0', '|||bad')):
            with self.assertRaises(ValueError):
                m2_sources(text)
        self.assertEqual(len(m2_sources(REF)), 1)

    def test_tokenization_mismatch_rejected_before_scoring(self):
        with self.assertRaisesRegex(ValueError, 'tokenization'):
            compare_m2(REF.replace('He go home', 'They go home'), REF)


@unittest.skipUnless(HAS_ERRANT, 'Install requirements-evaluation-gec.txt')
class ErrantM2IntegrationTests(unittest.TestCase):
    def test_real_official_cli_perfect_missed_wrong_edits(self):
        self.assertEqual(compare_m2(REF, REF)[0]['f0_5'], 1)
        missed = compare_m2(NOOP, REF)[0]
        self.assertEqual(missed['fn'], 1)
        self.assertEqual(missed['f0_5'], 0)
        wrong = compare_m2(WRONG, REF)[0]
        self.assertEqual((wrong['tp'], wrong['fp'], wrong['fn']), (0, 1, 1))

    def test_m2_multireference_uses_official_selection(self):
        reference = REF.rstrip() + '\nA 1 2|||R:VERB:SVA|||went|||REQUIRED|||-NONE-|||1\n\n'
        result = compare_m2(WRONG, reference)[0]
        self.assertEqual(result['f0_5'], 1)

    def test_file_outputs_and_existing_directory_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); ref = root / 'r.m2'; hyp = root / 'h.m2'
            ref.write_text(REF); hyp.write_text(REF)
            report = run_m2(ref, hyp, root / 'out', 'fixture', 'debug')
            self.assertEqual(report['units'], 1)
            self.assertEqual(report['errant']['f0_5'], 1)
            with self.assertRaises(ValueError):
                run_m2(ref, hyp, root / 'out', 'fixture', 'debug')


@unittest.skipUnless(HAS_SPACY_MODEL and HAS_ERRANT, 'Install ERRANT and en_core_web_sm')
class ParallelCorrectionIntegrationTests(unittest.TestCase):
    def test_whole_text_errant_and_failure_population(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); gold = root / 'g.jsonl'; pred = root / 'p.jsonl'
            rows = [{'id': 'wi:test:a0', 'source': 'WI_LOCNESS', 'task': 'grammar_correction', 'split': 'debug',
                     'messages': [{'role': 'user', 'content': TASKS['correction'][3] + 'He go home.'},
                                  {'role': 'assistant', 'content': '{"corrected_text":"He goes home."}'}]},
                    {'id': 'wi:missing:a0', 'source': 'WI_LOCNESS', 'task': 'grammar_correction', 'split': 'debug',
                     'input': 'I is here.', 'output': {'corrected_text': 'I am here.'}}]
            gold.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            pred.write_text(json.dumps({'id': 'wi:test:a0', 'raw_output': '{"corrected_text":"He goes home."}'}) + '\n')
            report = run_parallel(gold, pred, root / 'out', 'fixture', 'debug')
            self.assertEqual(report['coverage'], 0.5)
            self.assertEqual(report['errant']['f0_5'], 1)
            self.assertFalse(report['policy']['official_bea_result'])
            self.assertEqual(len((root / 'out/failures.jsonl').read_text().splitlines()), 1)

    def test_multiple_annotator_rows_are_not_silently_double_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); gold = root / 'g.jsonl'; pred = root / 'p.jsonl'
            rows = [{'id': 'wi:same:a' + str(i), 'source': 'WI_LOCNESS', 'task': 'grammar_correction',
                     'split': 'debug', 'input': 'He go home.', 'output': {'corrected_text': 'He goes home.'}} for i in range(2)]
            gold.write_text(''.join(json.dumps(row) + '\n' for row in rows)); pred.write_text('')
            with self.assertRaisesRegex(ValueError, 'Multiple annotators'):
                run_parallel(gold, pred, root / 'out', 'fixture', 'debug')
