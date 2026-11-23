"""Refuse cross-site state-changing requests explicitly.

Before this guard a cross-site write was refused only indirectly: the API accepts JSON bodies only,
and a browser must preflight a cross-origin JSON request, which the server never approves (it sends
no CORS headers). This guard makes the protection explicit for every request whose method is not
``GET``, ``HEAD`` or ``OPTIONS``:

* a ``Sec-Fetch-Site`` header other than ``same-origin`` or ``none`` is refused;
* an ``Origin`` header is refused unless it is ``http://`` on a trusted local host (``localhost``,
  ``127.0.0.1`` or ``[::1]``) with the port this server is serving on;
* a request with neither header (the CLI, the test client, ``curl``) is allowed, because the threat
  is a page in a browser, and browsers send these headers.

A refusal uses the ordinary error envelope: HTTP 403, code ``CROSS_ORIGIN_REFUSED``.
"""

from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from mcda_studio.api.errors import body

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
TRUSTED_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
ALLOWED_FETCH_SITES = frozenset({"same-origin", "none"})
MESSAGE = (
    "Refused a request from another website: this local server accepts changes only from its "
    "own pages and from programs that are not web browsers."
)


def _serving_port(scope: Scope, headers: Headers) -> int | None:
    server = scope.get("server")
    if server and server[1] is not None:
        return int(server[1])
    host = headers.get("host")
    if host is None:
        return None
    try:
        return urlsplit(f"http://{host}").port or 80
    except ValueError:
        return None


def _trusted_origin(origin: str, port: int | None) -> bool:
    try:
        parts = urlsplit(origin)
        origin_port = parts.port or (80 if parts.scheme == "http" else None)
    except ValueError:
        return False
    return (
        parts.scheme == "http"
        and parts.hostname in TRUSTED_HOSTS
        and port is not None
        and origin_port == port
        and parts.path in ("", "/")
        and not parts.username
        and not parts.password
    )


def refusal(scope: Scope) -> str | None:
    """Why this request must be refused, or None. Exposed for tests."""
    if scope["type"] != "http" or scope["method"] in SAFE_METHODS:
        return None
    headers = Headers(scope=scope)
    fetch_site = headers.get("sec-fetch-site")
    if fetch_site is not None and fetch_site.strip().lower() not in ALLOWED_FETCH_SITES:
        return f"Sec-Fetch-Site is {fetch_site!r}"
    origin = headers.get("origin")
    if origin is not None and not _trusted_origin(origin.strip(), _serving_port(scope, headers)):
        return f"Origin {origin!r} is not this server"
    return None


class CrossOriginGuard:
    """Pure ASGI middleware, so it adds nothing to responses it lets through."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        reason = refusal(scope)
        if reason is None:
            await self.app(scope, receive, send)
            return
        details = [{"path": "headers", "message": reason}]
        response = JSONResponse(body("CROSS_ORIGIN_REFUSED", MESSAGE, details), status_code=403)
        await response(scope, receive, send)
