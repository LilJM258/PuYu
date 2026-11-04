"""Create authored LEAF/ELLIPSE/GEC fixtures, not dataset-derived labels."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference.schema import DIMENSIONS, TaskOutputs, validate_tasks


def main():
    essay1 = ("Many students prefers online learning because it is convenience. "
        "They can study at home and saves time. However, online classes also have problems. "
        "For example, some students feels lonely and find it hard to ask questions. "
        "I think schools should combine online lessons with classroom activities, "
        "so students can enjoy flexibility and learn from each other.")
    essay2 = ("Our school should build a small garden. Students can grow vegetables and learn about nature. "
        "The garden is very good for students. It can also give us a quiet place to read. "
        "Although maintaining it takes time, each class could look after it for one week. "
        "This shared responsibility would make the project easier to manage.")
    essay3 = ("A school library should be a place where students can explore ideas at their own pace. "
        "For example, a student interested in the ocean could read a science book and then compare it with a travel memoir. "
        "Teachers can support this exploration by recommending books without deciding every choice. "
        "In this way, the library encourages both independent thinking and a lasting interest in reading.")
    definitions = [
        ("样例 1 · 在线学习 / 多处错误", essay1, [3.5, 3, 3, 3, 2.5, 4],
         "Your position is clear, and you consider both benefits and problems of online learning. "
         "Change 'students prefers' to 'students prefer' and 'students feels' to 'students feel' because the subjects are plural. "
         "Use the adjective 'convenient' after 'it is'. In 'can study at home and saves time', "
         "both verbs after 'can' need the base form, so use 'save'. Add a concrete example of how blended lessons would work.",
         essay1.replace("students prefers", "students prefer").replace("it is convenience", "it is convenient")
               .replace("and saves time", "and save time").replace("students feels", "students feel")),
        ("样例 2 · 校园花园 / 可选改进", essay2, [3.5, 3.5, 3, 3, 4, 4],
         "You explain the benefits of a school garden and offer a practical way to share the work. "
         "The sentence 'The garden is very good for students' is grammatical, but the benefit is vague. "
         "You could explain that growing vegetables gives students practical opportunities to learn about plants. "
         "This is an optional improvement in specificity, not a grammar correction.", essay2),
        ("样例 3 · 图书馆 / 无需改动", essay3, [4, 4, 4, 4, 4, 4.5],
         "The paragraph presents a clear purpose for the library and supports it with a relevant example. "
         "The final sentence connects the example to independent thinking and reading motivation. "
         "These links make the paragraph easy to follow. No specific grammatical correction is proposed in this feedback.", essay3),
    ]
    items = []
    annotation_specs = [
        [("students prefers", "error", "复数主语 students 应搭配 prefer。"),
         ("it is convenience", "error", "be 动词后此处需要形容词 convenient。"),
         ("and saves time", "error", "can 后并列动词使用原形 save。"),
         ("However, online classes also have problems.", "strength", "使用转折句，引出线上学习的另一面。"),
         ("students feels", "error", "复数主语 students 应搭配 feel。"),
         ("I think schools should combine online lessons with classroom activities,", "improvement", "可补充混合教学的具体例子，使建议更有说服力。")],
        [("The garden is very good for students.", "improvement", "语法正确，但 very good 较笼统，可以具体说明学习收益。"),
         ("This shared responsibility would make the project easier to manage.", "strength", "总结轮流照料的作用，回应维护花园的难点。")],
        [("For example, a student interested in the ocean could read a science book and then compare it with a travel memoir.", "strength", "以具体阅读情境支撑中心观点。"),
         ("In this way, the library encourages both independent thinking and a lasting interest in reading.", "strength", "结尾回扣自主思考和阅读兴趣。")],
    ]
    for i, (label, essay, scores, feedback, corrected) in enumerate(definitions, 1):
        responses = {"scoring": {"scores": dict(zip(DIMENSIONS, scores))},
                     "feedback": {"feedback": feedback}, "correction": {"corrected_text": corrected}}
        accepted, errors = validate_tasks(responses)
        assert not errors
        annotations = []
        for quote, category, note in annotation_specs[i - 1]:
            start = essay.index(quote)
            annotations.append(dict(start=start, end=start + len(quote), category=category, note=note))
        items.append(dict(label=label, sample_id=f"mock_00{i}", essay=essay, is_mock=True,
                          annotations=annotations, responses=accepted))
    for path, obj in ((ROOT / "data/mock/examples.json", items),
                      (ROOT / "configs/web_report.schema.json", TaskOutputs.model_json_schema())):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
