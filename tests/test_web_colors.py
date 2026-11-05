import unittest
from xml.etree import ElementTree

from inference.backend import examples
from web.app import original_preview
from web.presentation import annotated_original, corrected_text_html


class ColorRenderingTests(unittest.TestCase):
    def test_authored_colors_and_original_text_preserved(self):
        for item, counts in zip(examples(), [(4, 1, 1), (0, 1, 1), (0, 0, 2)]):
            with self.subTest(sample=item["sample_id"]):
                root = ElementTree.fromstring(original_preview(item["essay"]))
                self.assertEqual("".join(root.find("div").itertext()), item["essay"])
                for category, count in zip(("error", "improvement", "strength"), counts):
                    self.assertEqual(len(root.findall(f".//mark[@class='semantic-mark annotation-{category}']")), count)

    def test_edited_input_has_no_old_annotations_and_html_is_escaped(self):
        essay = '<script>alert("test")</script> & a new essay.'
        html = original_preview(essay)
        root = ElementTree.fromstring(html)
        self.assertEqual("".join(root.find("div").itertext()), essay)
        self.assertFalse(root.findall(".//mark"))
        self.assertNotIn("<script>", html)

    def test_overlapping_authored_annotations_rejected(self):
        annotations = [dict(start=0, end=3, category="error", note="first"),
                       dict(start=2, end=4, category="strength", note="overlap")]
        with self.assertRaises(ValueError):
            annotated_original("A sentence.", annotations)

    def test_only_changed_sentences_blue_and_corrected_text_preserved(self):
        item = examples()[0]
        corrected = item["responses"]["correction"]["corrected_text"]
        root = ElementTree.fromstring(corrected_text_html(item["essay"], corrected))
        marks = root.findall(".//mark")
        self.assertEqual(len(marks), 3)
        self.assertTrue(marks[0].text.startswith("Many students prefer"))
        self.assertNotIn("However, online classes also have problems.", [m.text for m in marks])
        self.assertEqual("".join(root.find("div").itertext()), corrected)

    def test_insertions_and_deletions_do_not_color_unchanged_neighbors(self):
        inserted = ElementTree.fromstring(corrected_text_html("One. Last.", "One. New! Last."))
        self.assertEqual([m.text for m in inserted.findall(".//mark")], ["New!"])
        deleted = ElementTree.fromstring(corrected_text_html("One. Removed. Last.", "One. Last."))
        self.assertFalse(deleted.findall(".//mark"))
        self.assertIn("被删除", deleted.find("p").text)

    def test_unchanged_whitespace_and_unsafe_correction(self):
        for before, after in (("Same.", "Same."), ("Same. Next.", "Same.\nNext.")):
            self.assertFalse(ElementTree.fromstring(corrected_text_html(before, after)).findall(".//mark"))
        text = '😀 <img src=x onerror="alert(1)"> changed.'
        html = corrected_text_html("Original.", text)
        self.assertNotIn("<img", html)
        self.assertEqual("".join(ElementTree.fromstring(html).find("div").itertext()), text)


if __name__ == "__main__":
    unittest.main()
