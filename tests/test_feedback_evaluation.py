import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.feedback import run, similarity
from evaluation.feedback_review import prepare, summarize, RATINGS
from evaluation.text_io import TASKS, load_gold, extract_text

HAS_METRICS = importlib.util.find_spec('rouge_score') is not None and importlib.util.find_spec('sacrebleu') is not None


def gold(key='leaf:test', text='Your example is clear. Add evidence to support the final claim.'):
    return {'id': key, 'task': 'essay_feedback', 'source': 'LEAF', 'split': 'debug',
            'messages': [{'role': 'user', 'content': TASKS['feedback'][3] + 'Students need a quiet library.'},
                         {'role': 'assistant', 'content': json.dumps({'feedback': text})}]}


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


class TextInputTests(unittest.TestCase):
    def test_raw_output_wins_and_no_schema_repair(self):
        text, reason = extract_text([{'row': {'feedback': 'repaired', 'raw_output': '{bad'}}], 'feedback')
        self.assertEqual(reason, 'invalid_model_json')
        self.assertEqual(extract_text([{'row': {'raw_output': '{"feedback":"ok","scores":{}}'}}], 'feedback')[1], 'unexpected_task_fields')
        self.assertEqual(extract_text([{'row': {'feedback': '  '}}], 'feedback')[1], 'empty_text')
        self.assertEqual(extract_text([{'row': {'feedback': 'ok'}}] * 2, 'feedback')[1], 'duplicate_prediction_id')
        self.assertEqual(extract_text([{'row': {'status': 'error', 'feedback': 'fake'}}], 'feedback')[1], 'generation_error')

    def test_gold_isolated_by_task_and_split_and_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'g.jsonl'
            row = gold()
            write_rows(p, [row])
            self.assertEqual(load_gold(p, 'feedback', 'debug')[row['id']]['reference'], json.loads(row['messages'][1]['content'])['feedback'])
            with self.assertRaises(ValueError):
                load_gold(p, 'feedback', 'test')
            row['messages'][0]['content'] = 'Unknown prefix\n\nText'
            write_rows(p, [row])
            with self.assertRaisesRegex(ValueError, 'instruction'):
                load_gold(p, 'feedback', 'debug')

    def test_missing_task_does_not_fall_back_to_other_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'g.jsonl'
            write_rows(p, [gold()])
            with self.assertRaisesRegex(ValueError, 'No WI_LOCNESS'):
                load_gold(p, 'correction', 'debug')


@unittest.skipUnless(HAS_METRICS, 'Install requirements-evaluation-feedback.txt')
class FeedbackReferenceTests(unittest.TestCase):
    def test_real_rouge_bleu_and_original_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); g = root / 'g.jsonl'; p = root / 'p.jsonl'
            row = gold(); text = json.loads(row['messages'][1]['content'])['feedback']
            write_rows(g, [row, gold('leaf:missing')])
            write_rows(p, [{'id': row['id'], 'raw_output': json.dumps({'feedback': text})}, {'id': 'extra', 'feedback': 'unknown'}])
            report = run(g, p, root / 'out', 'fixture', 'debug')
            self.assertEqual(report['coverage'], 0.5)
            self.assertEqual(report['metrics']['rouge']['rougeL']['f1'], 1)
            self.assertAlmostEqual(report['metrics']['bleu']['score_0_100'], 100)
            self.assertEqual(report['unknown_prediction_ids'], 1)
            self.assertIsNone(report['metrics']['bertscore'])
            self.assertEqual(len((root / 'out/failures.jsonl').read_text().splitlines()), 1)
            with self.assertRaises(ValueError):
                run(g, p, root / 'out', 'fixture', 'debug')

    def test_all_failed_has_no_fake_content_score(self):
        self.assertIsNone(similarity([{'valid': False}])['rouge'])


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.g = self.root / 'g.jsonl'; self.p = self.root / 'p.jsonl'
        write_rows(self.g, [gold()]); write_rows(self.p, [{'id': 'leaf:test', 'feedback': 'Add an example.'}])
        prepare(self.g, {'B': self.p}, self.root / 'packet', 'debug')
        self.manifest = self.root / 'packet/private_manifest.json'
        self.row = json.loads((self.root / 'packet/ratings_template.jsonl').read_text())

    def rating(self, who, score):
        return {**copy.deepcopy(self.row), 'rater_id': who, 'rated_at': '2026-11-09T12:00:00+08:00',
                **dict.fromkeys(RATINGS, score), 'evidence': 'Feedback asks for an example.',
                'hallucinated_error': False}

    def test_blind_packets_do_not_include_model_metadata(self):
        write_rows(self.p, [{'id': 'leaf:test', 'model': 'secret-model', 'feedback': 'Add an example.'}])
        prepare(self.g, {'secret-model': self.p}, self.root / 'blind', 'debug')
        public = (self.root / 'blind/review_packets.jsonl').read_text()
        self.assertNotIn('secret-model', public)
        self.assertIn('secret-model', (self.root / 'blind/private_manifest.json').read_text())

    def test_uncertain_claims_are_not_counted_as_verified(self):
        row = self.rating('one', 3)
        row['claim_judgments'] = [{'judgment': 'uncertain', 'evidence': 'Possible issue.', 'reason': 'Needs checking.'}]
        path = self.root / 'ratings.jsonl'; write_rows(path, [row])
        with self.assertRaisesRegex(ValueError, 'uncertain'):
            summarize(self.manifest, [path], self.root / 'out')
        row['hallucinated_error'] = None; write_rows(path, [row])
        report = summarize(self.manifest, [path], self.root / 'out')
        self.assertIsNone(report['groups'][0]['pooled_invalid_claim_rating_rate'])
        self.assertEqual(report['groups'][0]['checked_sample_ratings'], 0)

    def test_pending_and_noninvented_human_scores(self):
        path = self.root / 'pending.jsonl'; write_rows(path, [self.row])
        report = summarize(self.manifest, [path], self.root / 'out')
        self.assertEqual(report['pending_rows'], 1)
        self.assertEqual(report['groups'], [])

    def test_balanced_rater_means_and_agreement(self):
        path = self.root / 'ratings.jsonl'; write_rows(path, [self.rating('one', 3), self.rating('two', 5)])
        report = summarize(self.manifest, [path], self.root / 'out')
        group = report['groups'][0]
        self.assertEqual(group['dimensions'][RATINGS[0]]['mean'], 4)
        self.assertEqual(group['dimensions'][RATINGS[0]]['count'], 1)
        self.assertEqual(group['checked_sample_ratings'], 2)
        self.assertEqual(len((self.root / 'out/disagreements.jsonl').read_text().splitlines()), 4)

    def test_duplicate_reviews_and_changed_original_rejected(self):
        path = self.root / 'ratings.jsonl'; row = self.rating('one', 3)
        write_rows(path, [row, row])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            summarize(self.manifest, [path], self.root / 'out')
        row['raw_feedback'] = 'changed'; write_rows(path, [row])
        with self.assertRaisesRegex(ValueError, 'original'):
            summarize(self.manifest, [path], self.root / 'out')

    def test_cross_batch_reviews_rejected(self):
        path = self.root / 'ratings.jsonl'; row = self.rating('one', 3)
        row['review_manifest_fingerprint'] = 'other'; write_rows(path, [row])
        with self.assertRaisesRegex(ValueError, 'different prepared batch'):
            summarize(self.manifest, [path], self.root / 'out')

    def test_invalid_claims_and_n_a_explanation(self):
        row = self.rating('one', 3); row['edit_validity_1_5'] = None
        path = self.root / 'ratings.jsonl'; write_rows(path, [row])
        with self.assertRaisesRegex(ValueError, 'N/A'):
            summarize(self.manifest, [path], self.root / 'out')
        row['adjudication_notes'] = 'No specific edit proposed.'
        row['claim_judgments'] = [{'judgment': 'invalid', 'evidence': 'Invented mistake.', 'reason': 'Not in essay.'}]
        row['hallucinated_error'] = True; write_rows(path, [row])
        report = summarize(self.manifest, [path], self.root / 'out')
        self.assertEqual(report['groups'][0]['pooled_invalid_claim_rating_rate'], 1)
        self.assertIsNone(report['groups'][0]['dimensions']['edit_validity_1_5']['mean'])

    def test_llm_ratings_separate_and_require_provenance(self):
        row = self.rating('judge', 4); row['rater_kind'] = 'llm'
        path = self.root / 'ratings.jsonl'; write_rows(path, [row])
        with self.assertRaisesRegex(ValueError, 'judge_model'):
            summarize(self.manifest, [path], self.root / 'out')
        row.update(judge_model='fixture-judge', judge_prompt_version='rubric_v0.2_prompt_v1')
        write_rows(path, [row, self.rating('human', 3)])
        report = summarize(self.manifest, [path], self.root / 'out')
        self.assertEqual({g['rater_kind'] for g in report['groups']}, {'human', 'llm'})

    def test_no_score_invented_for_failed_generation(self):
        write_rows(self.p, [{'id': 'leaf:test', 'raw_output': '{bad'}])
        prepare(self.g, {'A': self.p}, self.root / 'failed_packet', 'debug')
        row = json.loads((self.root / 'failed_packet/ratings_template.jsonl').read_text())
        row.update(rater_id='one', rated_at='2026-11-09T12:00:00+08:00')
        row[RATINGS[0]] = 1
        path = self.root / 'bad.jsonl'; write_rows(path, [row])
        with self.assertRaisesRegex(ValueError, 'Invalid generated'):
            summarize(self.root / 'failed_packet/private_manifest.json', [path], self.root / 'out')
