import json
from io import BytesIO
from unittest.mock import Mock

import fitz
import pytest

from app import ServiceError, chunk_text, create_app

@pytest.fixture
def app(tmp_path):
    return create_app({'TESTING': True, 'DATA_DIR': str(tmp_path)})

@pytest.fixture
def client(app):
    return app.test_client()

def test_home(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'FinWise' in response.data

def test_chunks_preserve_content_and_boundaries():
    text = ' '.join('word' + str(i) for i in range(600))
    chunks = chunk_text(text)
    assert len(chunks) > 1
    assert all(len(x) <= 1100 for x in chunks)
    assert chunks[0].startswith('word0')
    assert chunks[-1].endswith('word599')
    assert all('word'+str(i) in ' '.join(chunks) for i in range(600))

@pytest.mark.parametrize('payload', [['invalid'], {}, {'question': ''}, {'question': 1}, {'question': 'x' * 2001}])
def test_invalid_question(client, payload):
    assert client.post('/api/chat', json=payload).status_code == 400

def test_upload_validation(client):
    assert client.post('/api/documents').status_code == 400
    assert client.post('/api/documents', data={'file': (BytesIO(b'not pdf'), 'fake.pdf')}).status_code == 400
    assert client.post('/api/documents', data={'file': (BytesIO(b'%PDF-broken'), 'fake.pdf')}).status_code == 400

def test_no_documents_is_explicit(app, client):
    kb = app.extensions['knowledge']
    kb._collection = Mock()
    kb._collection.count.return_value = 0
    data = client.post('/api/chat', json={'question': 'What is inflation?'}).get_json()
    assert data['sources'] == []
    assert 'not enough information' in data['answer']

def test_real_pdf_pages_embeddings_and_storage(app, client, monkeypatch):
    kb = app.extensions['knowledge']
    monkeypatch.setattr(kb, 'embed', lambda texts: [[1., 0., 0.] for _ in texts])
    pdf = fitz.open()
    for text in ['Budgeting tracks income and expenses.', 'Emergency funds provide accessible savings.']:
        page = pdf.new_page()
        page.insert_text((72,72), text)
    raw = pdf.tobytes()
    pdf.close()
    response = client.post('/api/documents', data={'file': (BytesIO(raw), '../../finance.pdf')})
    assert response.status_code == 201
    doc = response.get_json()
    assert doc['name'] == 'finance.pdf'
    assert doc['pages'] == 2
    result = kb.collection.get(include=['metadatas', 'documents'])
    assert {m['page'] for m in result['metadatas']} == {1,2}
    assert client.get('/documents/'+doc['id']).data == raw
    assert len(client.get('/api/documents').get_json()['documents']) == 1
    assert client.delete('/api/documents/'+doc['id']).status_code == 200
    assert kb.collection.count() == 0
    assert client.get('/documents/'+doc['id']).status_code == 404

PASSAGE = 'A budget tracks income and expenses.'
META = {'filename': 'notes.pdf', 'page': 2, 'pages': 3, 'document_id': 'a' * 32}
VALID = {'paragraphs': [{'text': 'A budget tracks income and expenses.',
                          'source_id': 1, 'quote': PASSAGE}]}


def grounded_kb(app, monkeypatch, content=VALID, distance=0.1):
    kb = app.extensions['knowledge']
    kb._collection = Mock()
    kb._collection.count.return_value = 1
    kb._collection.query.return_value = {
        'ids': [['chunk1']], 'documents': [[PASSAGE]],
        'metadatas': [[META]], 'distances': [[distance]]}
    kb._collection.get.return_value = {
        'ids': ['chunk1'], 'documents': [PASSAGE], 'metadatas': [META]}
    monkeypatch.setattr(kb, 'embed', lambda texts: [[1., 0.]])
    generate = Mock(return_value={'message': {'content': json.dumps(content)}})
    monkeypatch.setattr(kb, 'ollama', generate)
    return kb


def test_answer_cites_actual_retrieved_page(app, client, monkeypatch):
    grounded_kb(app, monkeypatch)
    response = client.post('/api/chat', json={'question': 'What is a budget?'})
    assert response.status_code == 200
    data = response.get_json()
    assert data['answer'].endswith('[1]')
    assert data['sources'][0]['page'] == 2
    assert data['sources'][0]['filename'] == 'notes.pdf'
    assert '#page=2' in data['sources'][0]['url']
    assert data['sources'][0]['evidence'] == [PASSAGE]


@pytest.mark.parametrize('content', [
    None, [], {}, {'paragraphs': None}, {'paragraphs': [{}]},
    {'answer': 'Budget [1]', 'citations': [1]},
    {'paragraphs': [{'text': 'Budget', 'source_id': 99, 'quote': PASSAGE}]},
    {'paragraphs': [{'text': 'Budget', 'source_id': 2, 'quote': PASSAGE}]},
    {'paragraphs': [{'text': 'Budget', 'source_id': '1', 'quote': PASSAGE}]},
    {'paragraphs': [{'text': 'Budget', 'source_id': True, 'quote': PASSAGE}]},
    {'paragraphs': [{'text': None, 'source_id': 1, 'quote': PASSAGE}]},
    {'paragraphs': [{'text': 'Budget [9]', 'source_id': 1, 'quote': PASSAGE}]},
    {'paragraphs': [{'text': 'Budget', 'source_id': 1, 'quote': 'Invented quote that is absent.'}]},
    {'paragraphs': [{'text': 'Budget', 'source_id': 1, 'quote': 'budget'}]},
    {'paragraphs': [{'text': 'Budget', 'source_id': 1, 'quote': None}]},
    {'paragraphs': [{'text': 'Budget', 'source_id': 1, 'quote': PASSAGE, 'page': 99}]},
    {'paragraphs': [{'text': 'Budget.\n\nUncited paragraph.', 'source_id': 1, 'quote': PASSAGE}]},
])
def test_invalid_citations_and_unsupported_quotes_fail_closed(app, client, monkeypatch, content):
    kb = grounded_kb(app, monkeypatch, content)
    response = client.post('/api/chat', json={'question': 'Budget?'})
    assert response.status_code == 503
    assert 'after two attempts' in response.get_json()['error']
    assert 'sources' not in response.get_json()
    assert kb.ollama.call_count == 2


def test_insufficient_evidence_has_no_sources(app, client, monkeypatch):
    kb = grounded_kb(app, monkeypatch, {'paragraphs': []})
    response = client.post('/api/chat', json={'question': 'What is quantum gravity?'})
    assert response.status_code == 200
    data = response.get_json()
    assert data['sources'] == []
    assert 'not contain enough information' in data['answer']
    assert kb.ollama.call_count == 1


def test_retry_has_failed_output_and_uses_verified_metadata(app, client, monkeypatch):
    kb = grounded_kb(app, monkeypatch)
    invalid = json.dumps({'paragraphs': [{'text': 'Budget', 'source_id': 2, 'quote': PASSAGE}]})
    generate = Mock(side_effect=[{'message': {'content': invalid}},
                                 {'message': {'content': json.dumps(VALID)}}])
    monkeypatch.setattr(kb, 'ollama', generate)
    response = client.post('/api/chat', json={'question': 'Budget?'})
    assert response.status_code == 200
    data = response.get_json()
    assert data['sources'][0]['label'] == 1
    assert data['sources'][0]['page'] == 2
    payload = generate.call_args.args[1]
    assert payload['format']['properties']['paragraphs']['items']['properties']['source_id']['enum'] == [1]
    assert any(m['role'] == 'assistant' and m['content'] == invalid for m in payload['messages'])
    kb._collection.query.assert_called_once()


def test_distance_cutoff_does_not_discard_relevant_passage(app, client, monkeypatch):
    kb = grounded_kb(app, monkeypatch, distance=0.71)
    response = client.post('/api/chat', json={'question': 'What is a budget?'})
    assert response.status_code == 200
    assert response.get_json()['sources'][0]['page'] == 2
    assert kb.ollama.call_count == 1


def test_lexical_retrieval_recovers_passage_missing_from_vector_results(app, monkeypatch):
    kb = grounded_kb(app, monkeypatch)
    kb._collection.count.return_value = 2
    kb._collection.get.return_value = {
        'ids': ['chunk1', 'chunk2'], 'documents': [PASSAGE, 'Valuation discounts future cash flows.'],
        'metadatas': [META, dict(META, page=3)]}
    sources = kb.retrieve('what is valuation', limit=2)
    assert any(s['page'] == 3 and 'Valuation' in s['text'] for s in sources)


def test_lexical_ranking_ignores_question_stopwords():
    from app import lexical_rank
    assert lexical_rank('what is valuation', ['a', 'b'],
                        ['What is the topic?', 'Valuation discounts future cash flows.']) == ['b']
    assert lexical_rank('quantum physics', ['a'], [PASSAGE]) == []


def test_quotes_must_match_selected_source_not_another_passage(app, client, monkeypatch):
    content = {'paragraphs': [{'text': 'Debt explanation', 'source_id': 1,
                              'quote': 'Debt requires repayment with interest.'}]}
    kb = grounded_kb(app, monkeypatch, content)
    kb._collection.count.return_value = 2
    kb._collection.get.return_value = {
        'ids': ['chunk1', 'chunk2'], 'documents': [PASSAGE, 'Debt requires repayment with interest.'],
        'metadatas': [META, dict(META, page=3)]}
    kb._collection.query.return_value['ids'] = [['chunk1', 'chunk2']]
    assert client.post('/api/chat', json={'question': 'Budget?'}).status_code == 503


@pytest.mark.parametrize('raw', ['not json', '[]', '{"paragraphs":'])
def test_invalid_json_retried_then_insufficient(app, client, monkeypatch, raw):
    kb = grounded_kb(app, monkeypatch)
    kb.ollama.side_effect = [{'message': {'content': raw}},
                            {'message': {'content': '{"paragraphs": []}'}}]
    response = client.post('/api/chat', json={'question': 'Budget?'})
    assert response.status_code == 200
    assert response.get_json()['sources'] == []
    assert kb.ollama.call_count == 2


def test_truncated_generation_not_accepted(app, client, monkeypatch):
    kb = grounded_kb(app, monkeypatch)
    kb.ollama.return_value = {'done_reason': 'length', 'message': {'content': json.dumps(VALID)}}
    assert client.post('/api/chat', json={'question': 'Budget?'}).status_code == 503


@pytest.mark.parametrize('page', [0, -1, True, '2', 99])
def test_invalid_page_metadata_never_becomes_citation(app, client, monkeypatch, page):
    kb = grounded_kb(app, monkeypatch)
    kb._collection.get.return_value['metadatas'] = [dict(META, page=page)]
    data = client.post('/api/chat', json={'question': 'Budget?'}).get_json()
    assert data['sources'] == []
    kb.ollama.assert_not_called()


def test_ollama_failure_is_actionable(app, client, monkeypatch):
    def fail(*args):
        raise ServiceError('Ollama unavailable: start Ollama.')
    monkeypatch.setattr(app.extensions['knowledge'], 'answer', fail)
    response = client.post('/api/chat', json={'question': 'Budget?'})
    assert response.status_code == 503
    assert 'Ollama' in response.get_json()['error']


def test_oversize_upload(client):
    assert client.post('/api/documents', data=b'x'*(20*1024*1024+1), content_type='application/pdf').status_code == 413


def test_document_path_validation(client):
    assert client.get('/documents/invalid').status_code == 404
    assert client.delete('/api/documents/invalid').status_code == 400


def test_slide_titles_and_contents_do_not_crowd_out_explanations(app, monkeypatch):
    kb = grounded_kb(app, monkeypatch)
    texts = ['Valuation', 'Contents Valuation • Income approach • DCF • Discounting overview',
             'Valuation estimates value by discounting expected future cash flows.']
    kb._collection.count.return_value = 3
    kb._collection.get.return_value = {
        'ids': ['title', 'contents', 'body'], 'documents': texts,
        'metadatas': [dict(META, page=p) for p in [1, 2, 3]]}
    kb._collection.query.return_value['ids'] = [['title', 'contents', 'body']]
    sources = kb.retrieve('what is valuation', limit=1)
    assert sources[0]['page'] == 3
