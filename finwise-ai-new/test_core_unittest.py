"""Dependency-free checks: python3 -m unittest discover -s tests -p '*unittest.py'."""
import ast
import re
import unittest
from pathlib import Path
from html.parser import HTMLParser

ROOT = Path(__file__).resolve().parents[1]
module = ast.parse((ROOT / 'app.py').read_text())
chunk_function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'chunk_text')
namespace = {'re': re}
exec(compile(ast.Module(body=[chunk_function], type_ignores=[]), 'app.py', 'exec'), namespace)
chunk_text = namespace['chunk_text']

class CoreTests(unittest.TestCase):
    def test_empty_text(self):
        self.assertEqual(chunk_text(' \n '), [])

    def test_page_content_is_retained(self):
        text = ' '.join('word'+str(i) for i in range(1000))
        chunks = chunk_text(text)
        self.assertTrue(all(len(c) <= 1100 for c in chunks))
        self.assertTrue(all('word'+str(i) in ' '.join(chunks) for i in range(1000)))
        self.assertTrue(chunks[-1].endswith('word999'))

    def test_unbroken_text_terminates(self):
        chunks = chunk_text('x'*5000)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 1100 for c in chunks))

    def test_short_page_remains_one_chunk(self):
        self.assertEqual(chunk_text('Income\n and   expenses'), ['Income and expenses'])

    def test_html_ids_unique_and_forms_labeled(self):
        class Parser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.ids = []
                self.labels = []
            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if 'id' in attrs:
                    self.ids.append(attrs['id'])
                if tag == 'label':
                    self.labels.append(attrs.get('for'))
        parser = Parser()
        parser.feed((ROOT / 'templates/index.html').read_text())
        self.assertEqual(len(parser.ids), len(set(parser.ids)))
        self.assertTrue(all(label in parser.ids for label in parser.labels))

if __name__ == '__main__':
    unittest.main()
