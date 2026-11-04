"""Safe HTML views for scores, authored annotations and sentence differences."""

import re
from difflib import SequenceMatcher
from html import escape

from inference.schema import DIMENSIONS

EMPTY_CARDS = '<div class="empty-report">未提供有效六维评分 · 不补造缺失分数</div>'
EMPTY_PREVIEW = '<div class="empty-report">载入样例后显示颜色标注；也可在上方输入自己的作文。</div>'
ANNOTATION_LABELS = {"error": "明确错误", "improvement": "可选改进", "strength": "表达亮点"}

COLOR_LEGEND = '''<aside class="color-legend" aria-label="颜色说明">
  <strong>颜色说明</strong><div class="legend-items">
  <span><i class="legend-swatch annotation-error"></i>红色：原文明确错误</span>
  <span><i class="legend-swatch annotation-improvement"></i>黄色：可选改进</span>
  <span><i class="legend-swatch annotation-strength"></i>绿色：表达亮点</span>
  <span><i class="legend-swatch revision-changed"></i>蓝色：纠正后有改动的句子</span></div>
  <p>原文三色为模拟样例的人工标注；蓝色仅表示文本变化，不代表对修改质量的判断。未标色不代表没有问题。</p>
</aside>'''


def annotated_original(essay: str, annotations: list[dict] | None = None) -> str:
    """Annotations are authored fixture metadata, never inferred from LEAF text."""
    if not essay:
        return EMPTY_PREVIEW
    chunks, cursor = [], 0
    for item in sorted(annotations or [], key=lambda item: item["start"]):
        start, end, category = item["start"], item["end"], item["category"]
        if category not in ANNOTATION_LABELS or not (cursor <= start < end <= len(essay)):
            raise ValueError("样例标注类别、边界或重叠区间无效")
        chunks.append(escape(essay[cursor:start]))
        title = ANNOTATION_LABELS[category] + "：" + item["note"]
        chunks.append(f'<mark class="semantic-mark annotation-{category}" title="{escape(title, quote=True)}">'
                      f'{escape(essay[start:end])}</mark>')
        cursor = end
    chunks.append(escape(essay[cursor:]))
    note = "样例人工标注 · 鼠标移到色块上可查看原因。" if annotations else "当前作文没有人工颜色标注；不自动推断错误、改进点或亮点。"
    return f'<article class="annotated-panel"><p class="annotation-note">{note}</p><div class="annotated-body">' + "".join(chunks) + '</div></article>'


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Approximate English sentence boundaries using final punctuation/newlines."""
    boundaries = [match.end() for match in re.finditer(r'''[.!?]+["'”’)]*(?=\s|$)|\n+''', text)]
    if not boundaries or boundaries[-1] != len(text):
        boundaries.append(len(text))
    spans, start = [], 0
    for end in boundaries:
        left = start
        while left < end and text[left].isspace():
            left += 1
        right = end
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            spans.append((left, right))
        start = end
    return spans


def corrected_text_html(original: str, corrected: str) -> str:
    if not corrected:
        return ""
    before, after = sentence_spans(original), sentence_spans(corrected)
    matcher = SequenceMatcher(None, [original[a:b] for a, b in before],
                              [corrected[a:b] for a, b in after], autojunk=False)
    changed, deleted = set(), 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "insert"):
            changed.update(range(j1, j2))
        elif tag == "delete":
            deleted += i2 - i1
    chunks, cursor = [], 0
    for index, (start, end) in enumerate(after):
        chunks.append(escape(corrected[cursor:start]))
        sentence = escape(corrected[start:end])
        if index in changed:
            sentence = '<mark class="revision-changed" title="此句相较原文有改动">' + sentence + '</mark>'
        chunks.append(sentence)
        cursor = end
    chunks.append(escape(corrected[cursor:]))
    if original == corrected:
        note = "与原文一致，本次没有文本改动。"
    elif not changed and not deleted:
        note = "仅空白或段落布局发生变化，没有标记句子内容修改。"
    else:
        note = f"蓝底标出了 {len(changed)} 个有改动或新增的句子。"
        if deleted:
            note += f"另有 {deleted} 个原文句子被删除，请对照原文查看。"
    return '<article class="annotated-panel"><p class="annotation-note">' + note + '</p><div class="annotated-body">' + "".join(chunks) + '</div></article>'


def score_cards(scoring: dict | None) -> str:
    if scoring is None:
        return EMPTY_CARDS
    cards = []
    for key, label in DIMENSIONS.items():
        score = scoring["scores"][key]
        cards.append(
            f'<div class="score-card"><span>{label}</span><small>{key}</small>'
            f'<strong>{score:g}<em> / 5</em></strong>'
            f'<div class="score-track"><i style="width:{score / 5 * 100:g}%"></i></div></div>'
        )
    return '<div class="score-grid">' + "".join(cards) + (
        '</div><p class="score-note">六维范围 1–5，保留原始小数；'
        '不取整、不截断、不另造综合分。</p>'
    )
