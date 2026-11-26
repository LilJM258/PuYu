import json
import tempfile
import unittest
from pathlib import Path
from scripts.import_human_text import import_text


class HumanTextTests(unittest.TestCase):
    def files(self, folder, filled=True):
        row = {'output_id':'output_000001','sample_id':'sample1','output_valid':True,
               'review_manifest_fingerprint':'frozen','raw_feedback':'Explain why.'}
        template, text = folder/'ratings.jsonl', folder/'人工评分表.txt'
        template.write_text(json.dumps(row)+'\n',encoding='utf-8')
        (folder/'review_packets.jsonl').write_text(json.dumps(dict(row,essay='I agree.',feedback='Explain why.'))+'\n',encoding='utf-8')
        text.write_text('评价者编号：r1\n评价日期：2026-11-25T10:00:00+08:00\n===== output_000001 =====\n'
            '样本编号：sample1\n【作文】\nI agree.\n【反馈】\nExplain why.\n【仅填写以下栏目】\n'
            '问题识别准确性：'+('3' if filled else '____')+'\n建议具体性：4\n修改有效性：NA\n忠实于原文：5\n'
            '证据与理由：仅有总体建议\n不适用或分歧说明：没有具体改法\n逐项判断JSON：[]\n',encoding='utf-8')
        return template,text

    def test_import_preserves_fingerprint_and_na(self):
        with tempfile.TemporaryDirectory() as directory:
            template,text=self.files(Path(directory))
            row=import_text(template,text)[0]
        self.assertEqual(row['review_manifest_fingerprint'],'frozen')
        self.assertEqual(row['problem_accuracy_1_5'],3)
        self.assertIsNone(row['edit_validity_1_5'])

    def test_unfilled_form_is_not_automatically_scored(self):
        with tempfile.TemporaryDirectory() as directory:
            template,text=self.files(Path(directory),False)
            with self.assertRaisesRegex(ValueError,'尚未填写'):
                import_text(template,text)

    def test_modified_feedback_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            template,text=self.files(Path(directory))
            text.write_text(text.read_text(encoding='utf-8').replace('Explain why.','Different feedback.'),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'作文或反馈被修改'):
                import_text(template,text)

    def test_duplicate_output_id_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            template,text=self.files(Path(directory))
            text.write_text(text.read_text(encoding='utf-8')+'\n===== output_000001 =====\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'编号重复'):
                import_text(template,text)

    def test_explicit_invalid_claim_can_be_summarized(self):
        with tempfile.TemporaryDirectory() as directory:
            template,text=self.files(Path(directory))
            claim = [{'judgment':'invalid','evidence':'原文 I agree.','reason':'原文没有该错误'}]
            text.write_text(text.read_text(encoding='utf-8').replace('逐项判断JSON：[]',
                '逐项判断JSON：'+json.dumps(claim,ensure_ascii=False)),encoding='utf-8')
            row=import_text(template,text)[0]
        self.assertTrue(row['hallucinated_error'])
        self.assertEqual(row['claim_judgments'],claim)

    def test_uncertain_claim_does_not_invent_hallucination_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            template,text=self.files(Path(directory))
            claim = [{'judgment':'uncertain','evidence':'原文 I agree.','reason':'需要复核'}]
            text.write_text(text.read_text(encoding='utf-8').replace('逐项判断JSON：[]',
                '逐项判断JSON：'+json.dumps(claim)),encoding='utf-8')
            row=import_text(template,text)[0]
        self.assertIsNone(row['hallucinated_error'])

    def test_naive_date_rejected_before_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            template,text=self.files(Path(directory))
            text.write_text(text.read_text(encoding='utf-8').replace('2026-11-25T10:00:00+08:00',
                '2026-11-25'),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'带时区'):
                import_text(template,text)


if __name__=='__main__':
    unittest.main()
