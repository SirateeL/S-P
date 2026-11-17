"""Small-team access control; no changes to the document database."""
import ipaddress
import os
import threading
import time
from functools import wraps
from urllib.parse import urlsplit

from flask import g, jsonify, request
from werkzeug.security import check_password_hash


def configure_security(app):
    app.config.from_mapping(
        FINWISE_ENV=os.environ.get('FINWISE_ENV', 'local'),
        ADMIN_PASSWORD_HASH=os.environ.get('FINWISE_ADMIN_PASSWORD_HASH', ''),
        VIEWER_PASSWORD_HASH=os.environ.get('FINWISE_VIEWER_PASSWORD_HASH', ''),
        PUBLIC_ORIGIN=os.environ.get('FINWISE_PUBLIC_ORIGIN', ''),
    )


def install_security(app):
    mode = app.config['FINWISE_ENV']
    if mode not in {'local', 'production'}:
        raise RuntimeError('FINWISE_ENV must be local or production.')
    admin = app.config['ADMIN_PASSWORD_HASH']
    viewer = app.config['VIEWER_PASSWORD_HASH']
    if viewer and not admin:
        raise RuntimeError('A viewer password requires an administrator password.')
    for password_hash in (admin, viewer):
        if password_hash:
            if not password_hash.startswith(('scrypt:', 'pbkdf2:')) or len(password_hash.split('$')) != 3:
                raise RuntimeError('Use a Werkzeug-generated password hash, not a plaintext password.')
            try:
                check_password_hash(password_hash, 'configuration-check')
            except (ValueError, TypeError) as exc:
                raise RuntimeError('Invalid password hash configuration.') from exc
    origin = app.config['PUBLIC_ORIGIN'].rstrip('/')
    parsed = urlsplit(origin)
    if mode == 'production':
        if not admin or parsed.scheme != 'https' or not parsed.hostname or parsed.path or parsed.query or parsed.fragment or parsed.username:
            raise RuntimeError('Production requires FINWISE_ADMIN_PASSWORD_HASH and an HTTPS FINWISE_PUBLIC_ORIGIN.')
        app.config['TRUSTED_HOSTS'] = [parsed.hostname]
    else:
        app.config['TRUSTED_HOSTS'] = ['localhost', '127.0.0.1', '[::1]']

    # Bounded, process-local throttling for password guessing. Behind a proxy,
    # remote_addr is intentionally the proxy: never trust arbitrary X-Forwarded-For.
    failures = {}
    failures_lock = threading.Lock()

    @app.before_request
    def protect():
        # Access host here so Flask validates it even for unknown routes.
        request.host
        g.finwise_role = 'admin'
        if not admin:
            try:
                loopback = ipaddress.ip_address(request.remote_addr or '').is_loopback
            except ValueError:
                loopback = False
            if not loopback:
                return jsonify(error='Unauthenticated local mode accepts loopback connections only.'), 403
        else:
            now = time.monotonic()
            peer = request.remote_addr or 'unknown'
            with failures_lock:
                for key in list(failures):
                    if now - failures[key][0] >= 60:
                        del failures[key]
                blocked = failures.get(peer, (now, 0))[1] >= 10 or (peer not in failures and len(failures) >= 1024)
            if blocked:
                response = jsonify(error='Too many sign-in attempts. Please wait one minute.')
                response.status_code = 429
                response.headers['Retry-After'] = '60'
                return response
            auth = request.authorization
            role = None
            if auth and auth.type.lower() == 'basic' and isinstance(auth.password, str):
                candidate = viewer if auth.username == 'viewer' and viewer else admin
                valid = check_password_hash(candidate, auth.password)
                if valid and auth.username == 'admin':
                    role = 'admin'
                elif valid and auth.username == 'viewer' and viewer:
                    role = 'viewer'
            if role is None:
                with failures_lock:
                    started, count = failures.get(peer, (now, 0))
                    failures[peer] = (started, count + 1)
                response = jsonify(error='Sign in with your FinWise credentials.')
                response.status_code = 401
                response.headers['WWW-Authenticate'] = 'Basic realm="FinWise", charset="UTF-8"'
                return response
            g.finwise_role = role
        if request.method not in {'GET', 'HEAD', 'OPTIONS'}:
            # This API has no cross-origin clients. A custom header makes form
            # CSRF impossible and forces cross-origin JS through denied preflight.
            if request.headers.get('X-FinWise-Request') != '1':
                return jsonify(error='Missing same-origin request header. Reload FinWise and try again.'), 403
            expected = origin or request.host_url.rstrip('/')
            if (request.headers.get('Sec-Fetch-Site') == 'cross-site' or
                    request.headers.get('Origin', expected) != expected):
                return jsonify(error='Cross-origin requests are not allowed.'), 403
        if request.path.startswith('/api/documents') and request.method in {'POST', 'DELETE'} and g.finwise_role != 'admin':
            return jsonify(error='Only the administrator can upload or delete PDFs.'), 403

    @app.after_request
    def private_headers(response):
        response.headers['Cache-Control'] = 'private, no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Referrer-Policy'] = 'no-referrer'
        # Existing charts assign inline style properties; retain those styles.
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        return response


def single_ai_request(kb):
    """Bound expensive work without changing the retrieval/generation pipeline."""
    def decorate(function):
        @wraps(function)
        def guarded(*args, **kwargs):
            if not kb.ai_slot.acquire(blocking=False):
                response = jsonify(error='FinWise is processing another AI request. Please try again shortly.')
                response.status_code = 429
                response.headers['Retry-After'] = '5'
                return response
            try:
                return function(*args, **kwargs)
            finally:
                kb.ai_slot.release()
        return guarded
    return decorate
