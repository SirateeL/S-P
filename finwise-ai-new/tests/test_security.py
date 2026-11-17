"""Security checks use temporary storage, never the existing PDF library."""
from io import BytesIO
from unittest.mock import Mock

import pytest
from werkzeug.security import generate_password_hash

from app import create_app


@pytest.fixture
def protected(tmp_path):
    app = create_app({'TESTING': True, 'DATA_DIR': str(tmp_path),
                      'FINWISE_ENV': 'local', 'PUBLIC_ORIGIN': '',
                      'ADMIN_PASSWORD_HASH': generate_password_hash('admin-test', method='pbkdf2:sha256:1000'),
                      'VIEWER_PASSWORD_HASH': generate_password_hash('viewer-test', method='pbkdf2:sha256:1000')})
    kb = app.extensions['knowledge']
    kb._collection = Mock()
    kb._collection.get.return_value = {'metadatas': []}
    kb.answer = Mock(return_value={'answer': 'Insufficient evidence', 'sources': []})
    kb.ingest = Mock(return_value={'id': 'a' * 32, 'pages': 1})
    (kb.uploads / ('a' * 32 + '.pdf')).write_bytes(b'%PDF-test')
    return app, app.test_client(), kb


@pytest.mark.parametrize('path', ['/', '/static/app.js', '/api/status', '/api/documents', '/documents/' + 'a' * 32])
def test_anonymous_read_denied(protected, path):
    _, client, _ = protected
    response = client.get(path)
    assert response.status_code == 401
    assert response.headers['WWW-Authenticate'].startswith('Basic ')
    assert response.headers['Cache-Control'] == 'private, no-store'


@pytest.mark.parametrize('method,path', [('post', '/api/documents'), ('delete', '/api/documents/' + 'a' * 32), ('post', '/api/chat')])
def test_anonymous_write_denied_before_work(protected, method, path):
    _, client, kb = protected
    assert getattr(client, method)(path).status_code == 401
    kb.ingest.assert_not_called()
    kb.answer.assert_not_called()
    kb._collection.delete.assert_not_called()


@pytest.mark.parametrize('auth', [('admin', 'wrong'), ('viewer', 'admin-test'), ('unknown', 'admin-test')])
def test_wrong_credentials_denied(protected, auth):
    _, client, _ = protected
    assert client.get('/', auth=auth).status_code == 401


def test_viewer_can_read_pdf_and_ask_but_not_modify(protected):
    _, client, kb = protected
    auth = ('viewer', 'viewer-test')
    headers = {'X-FinWise-Request': '1'}
    response = client.get('/documents/' + 'a' * 32, auth=auth)
    assert response.status_code == 200
    assert response.data == b'%PDF-test'
    assert response.headers['X-Content-Type-Options'] == 'nosniff'
    assert client.get('/api/documents', auth=auth).status_code == 200
    assert client.post('/api/chat', auth=auth, headers=headers, json={'question': 'Budget?'}).status_code == 200
    assert client.post('/api/documents', auth=auth, headers=headers).status_code == 403
    assert client.delete('/api/documents/' + 'a' * 32, auth=auth, headers=headers).status_code == 403
    kb.ingest.assert_not_called()
    kb._collection.delete.assert_not_called()


def test_admin_upload_delete_still_work(protected):
    _, client, kb = protected
    auth, headers = ('admin', 'admin-test'), {'X-FinWise-Request': '1'}
    response = client.post('/api/documents', auth=auth, headers=headers,
                           data={'file': (BytesIO(b'%PDF-test'), 'notes.pdf')})
    assert response.status_code == 201
    kb.ingest.assert_called_once_with(b'%PDF-test', 'notes.pdf')
    assert client.delete('/api/documents/' + 'a' * 32, auth=auth, headers=headers).status_code == 200
    kb._collection.delete.assert_called_once_with(where={'document_id': 'a' * 32})


@pytest.mark.parametrize('headers', [{}, {'X-FinWise-Request': '0'},
    {'X-FinWise-Request': '1', 'Origin': 'https://evil.example'},
    {'X-FinWise-Request': '1', 'Origin': 'null'},
    {'X-FinWise-Request': '1', 'Sec-Fetch-Site': 'cross-site'}])
def test_csrf_rejected_before_upload_or_delete(protected, headers):
    _, client, kb = protected
    for method, path in [('post', '/api/documents'), ('delete', '/api/documents/' + 'a' * 32)]:
        response = getattr(client, method)(path, auth=('admin', 'admin-test'), headers=headers)
        assert response.status_code == 403
    kb.ingest.assert_not_called()
    kb._collection.delete.assert_not_called()


def test_cross_origin_preflight_not_authorized(protected):
    _, client, _ = protected
    response = client.options('/api/documents', auth=('admin', 'admin-test'), headers={
        'Origin': 'https://evil.example', 'Access-Control-Request-Headers': 'X-FinWise-Request',
        'Access-Control-Request-Method': 'POST'})
    assert 'Access-Control-Allow-Origin' not in response.headers
    assert 'Access-Control-Allow-Headers' not in response.headers


def test_busy_request_and_exception_release_slot(protected):
    _, client, kb = protected
    kwargs = {'auth': ('admin', 'admin-test'), 'headers': {'X-FinWise-Request': '1'}, 'json': {'question': 'Budget?'}}
    kb.ai_slot.acquire()
    try:
        response = client.post('/api/chat', **kwargs)
        assert response.status_code == 429
        assert response.headers['Retry-After'] == '5'
        kb.answer.assert_not_called()
    finally:
        kb.ai_slot.release()
    kb.answer.side_effect = ValueError('test failure')
    assert client.post('/api/chat', **kwargs).status_code == 400
    kb.answer.side_effect = None
    assert client.post('/api/chat', **kwargs).status_code == 200


@pytest.mark.parametrize('extra', [
    {'FINWISE_ENV': 'invalid'}, {'FINWISE_ENV': 'production'},
    {'FINWISE_ENV': 'production', 'ADMIN_PASSWORD_HASH': 'plaintext'},
    {'FINWISE_ENV': 'production', 'PUBLIC_ORIGIN': 'http://example.com'},
    {'VIEWER_PASSWORD_HASH': 'plaintext'},
])
def test_unsafe_config_fails_before_storage_creation(tmp_path, extra):
    path = tmp_path / 'must-not-be-created'
    config = {'DATA_DIR': str(path), 'FINWISE_ENV': 'local', 'ADMIN_PASSWORD_HASH': '',
              'VIEWER_PASSWORD_HASH': '', 'PUBLIC_ORIGIN': ''}
    config.update(extra)
    with pytest.raises(RuntimeError):
        create_app(config)
    assert not path.exists()


def test_production_origin_and_host_validation(tmp_path):
    app = create_app({'TESTING': True, 'DATA_DIR': str(tmp_path), 'FINWISE_ENV': 'production',
        'PUBLIC_ORIGIN': 'https://finwise.example.com', 'VIEWER_PASSWORD_HASH': '',
        'ADMIN_PASSWORD_HASH': generate_password_hash('test', method='pbkdf2:sha256:1000')})
    client = app.test_client()
    assert client.get('/', base_url='https://finwise.example.com', auth=('admin', 'test')).status_code == 200
    assert client.get('/', base_url='https://evil.example', auth=('admin', 'test')).status_code == 400
    app.extensions['knowledge'].answer = Mock(return_value={'answer': 'test', 'sources': []})
    assert client.post('/api/chat', base_url='https://finwise.example.com', auth=('admin', 'test'),
                       headers={'X-FinWise-Request': '1', 'Origin': 'https://finwise.example.com'},
                       json={'question': 'Budget?'}).status_code == 200


def test_local_without_password_remains_loopback_only(tmp_path):
    app = create_app({'TESTING': True, 'DATA_DIR': str(tmp_path), 'FINWISE_ENV': 'local',
                      'ADMIN_PASSWORD_HASH': '', 'VIEWER_PASSWORD_HASH': '', 'PUBLIC_ORIGIN': ''})
    client = app.test_client()
    assert client.get('/').status_code == 200
    assert client.get('/', environ_overrides={'REMOTE_ADDR': '203.0.113.1'}).status_code == 403
    assert client.get('/', base_url='http://evil.example').status_code == 400
    assert client.post('/api/documents').status_code == 403


def test_password_guessing_throttled(protected):
    _, client, _ = protected
    for _ in range(10):
        assert client.get('/', auth=('admin', 'wrong')).status_code == 401
    response = client.get('/', auth=('admin', 'wrong'))
    assert response.status_code == 429
    assert response.headers['Retry-After'] == '60'


def test_busy_upload_does_not_ingest(protected):
    _, client, kb = protected
    kb.ai_slot.acquire()
    try:
        assert client.post('/api/documents', auth=('admin', 'admin-test'),
                           headers={'X-FinWise-Request': '1'}).status_code == 429
        kb.ingest.assert_not_called()
    finally:
        kb.ai_slot.release()
