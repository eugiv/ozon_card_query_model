import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from tools.readable_docs import DOCUMENTS, render_document

ROOT = Path(__file__).resolve().parents[1]


class DocumentParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.links = []
        self.text = []
        self.charset = None

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            self.ids.append(values["id"])
        if tag == "a":
            self.links.append(values["href"])
        if tag == "meta" and "charset" in values:
            self.charset = values["charset"]

    def handle_data(self, data):
        self.text.append(data)


def test_renderer_preserves_cyrillic_tables_code_and_unique_navigation():
    source = '# Учебник\n\n## Начало\n\nПривет, **мир**.\n\n## Начало\n\n' \
             '| Поле | Значение |\n|---|---|\n| orders | 2 |\n\n```python\nprint("Привет")\n```'
    rendered = render_document(source)
    parsed = DocumentParser()
    parsed.feed(rendered)
    assert parsed.charset == 'utf-8'
    assert len(parsed.ids) == len(set(parsed.ids)) == 3
    assert '#начало' in parsed.links and '#начало-1' in parsed.links
    assert 'Привет' in ''.join(parsed.text)
    assert '<table>' in rendered and '<code class="language-python">' in rendered
    assert '<script' not in rendered


def test_renderer_escapes_raw_html_and_links_to_readable_siblings():
    rendered = render_document('# Текст\n\n<script>alert(1)</script>\n\n'
                               '[Ответы](answers_to_feedback.md)')
    assert '<script>' not in rendered
    assert 'href="answers_to_feedback.html"' in rendered
    with pytest.raises(ValueError):
        render_document('Текст без заголовка')


def test_contents_links_to_explanations_inside_each_chapter():
    rendered = render_document('# Учебник\n\n## Векторы\n\n'
                               '### Считаем по шагам\n\nПример.\n\n'
                               '### Проверяем себя\n\nОтвет.\n\n'
                               '## Метрики\n\n### Считаем по шагам\n\nПример.')
    contents = rendered.split('<summary>Оглавление</summary>', 1)[1].split('</details>', 1)[0]
    parsed = DocumentParser()
    parsed.feed(contents)
    assert parsed.links == ['#векторы', '#считаем-по-шагам', '#проверяем-себя',
                            '#метрики', '#считаем-по-шагам-1']
    assert contents.count('<ul>') == contents.count('</ul>') == 2


@pytest.mark.parametrize('name', DOCUMENTS)
def test_saved_documents_are_utf8_current_and_have_working_links(name):
    source = ROOT / 'docs' / name
    text = source.read_text(encoding='utf-8')
    target = source.with_suffix('.html')
    rendered = target.read_text(encoding='utf-8')
    assert rendered == render_document(text)
    assert '\ufffd' not in text and '\ufffd' not in rendered
    parsed = DocumentParser()
    parsed.feed(rendered)
    assert parsed.charset == 'utf-8'
    assert len(parsed.ids) == len(set(parsed.ids))
    assert re.search('[А-Яа-я]{5,}', ''.join(parsed.text))
    for link in parsed.links:
        path, _, anchor = link.partition('#')
        if path:
            assert (target.parent / path).is_file(), link
        elif anchor:
            assert anchor in parsed.ids, link


def test_manual_covers_the_whole_project_without_mixing_evaluations():
    text = (ROOT / 'docs/project_manual.md').read_text(encoding='utf-8')
    for fragment in ['TF-IDF', 'E5', 'LightGCN', 'CatBoost', 'RRF', 'PU', 'BPR',
                     'Recall@10', 'NDCG@10', 'bootstrap', 'patience', 'seed',
                     '477 579', '18 444', '19,06%', '22,85%', '17,7931%', '17,5354%',
                     'не рост продаж', '999', 'train', 'validation', 'refit',
                     'Сквозной пример', 'Самопроверка', 'Карта модулей', 'Карта ячеек',
                     'Математический букварь', 'Одно обновление одного веса целиком',
                     'Большой контрольный разбор', 'Подробное решение',
                     'Как найти оставшийся пробел']:
        assert fragment in text, fragment


def test_manual_chapters_are_continuous_and_subtopics_are_navigable():
    text = (ROOT / 'docs/project_manual.md').read_text(encoding='utf-8')
    chapters = re.findall(r'^## (\d+)\. ', text, re.MULTILINE)
    assert [int(number) for number in chapters] == list(range(1, 33))
    rendered = render_document(text)
    contents = rendered.split('<summary>Оглавление</summary>', 1)[1].split('</details>', 1)[0]
    parsed = DocumentParser()
    parsed.feed(contents)
    subtopics = re.findall(r'<h3 id="([^"]+)">', rendered)
    assert subtopics
    assert all(f'#{anchor}' in parsed.links for anchor in subtopics)


def test_manual_worked_case_matches_project_ranking_functions():
    import pandas as pd

    from dormant_learning import fusion_scores, ranking_metrics

    text = (ROOT / 'docs/project_manual.md').read_text(encoding='utf-8')
    example = text.split('### 31.4. ', 1)[1].split('### 31.5. ', 1)[0]
    rows = re.findall(r'^\| (q\d+) \| ([\d.]+) \| ([\d.]+) \|$', example, re.MULTILINE)
    assert len(rows) == 3
    candidates = pd.DataFrame(rows, columns=['normquery', 'embedding_mean', 'graph_score'])
    candidates['sku'] = 'А'
    for column in ['embedding_mean', 'graph_score']:
        candidates[column] = pd.to_numeric(candidates[column])
    candidates['score'] = fusion_scores(candidates, {'pretrained_embeddings': 0.5, 'co_sales': 0.5})
    declared_scores = dict(re.findall(r'`(q\d+)=[^`]*≈([\d.]+)`', example))
    assert set(declared_scores) == set(candidates['normquery'])
    for query, score in candidates.set_index('normquery')['score'].items():
        assert score == pytest.approx(float(declared_scores[query]), abs=5e-7)
    declared_order = re.search(r'Порядок \*\*(q\d+(?: → q\d+)+)\*\*', example)
    assert declared_order is not None
    assert candidates.sort_values('score', ascending=False)['normquery'].tolist() == declared_order.group(1).split(' → ')

    target_match = re.search(r'заказы были по \*\*([^*]+)\*\*', example)
    assert target_match is not None
    targets = target_match.group(1).split(', ')
    positives = pd.DataFrame({'sku': 'А', 'normquery': targets})
    metrics = ranking_metrics(candidates, positives, ks=(2,))
    for key, label in [('recall@2', 'Recall@2'), ('precision@2', 'Precision@2'),
                       ('retrieval_recall', 'retrieval recall')]:
        percentage_match = re.search(rf'{label}=`[^`]*[≈=]([\d.]+)%`', example)
        assert percentage_match is not None
        declared = percentage_match.group(1)
        assert 100 * metrics[key] == pytest.approx(float(declared), abs=0.005)
    ndcg_match = re.search(r'NDCG≈\*\*([\d.]+)\*\*', example)
    assert ndcg_match is not None
    declared_ndcg = float(ndcg_match.group(1))
    assert metrics['ndcg@2'] == pytest.approx(declared_ndcg, abs=5e-6)
    assert metrics['positive_pairs'] == 3
    assert set(targets) - set(candidates['normquery']) == {'q7'}