# Changelog

## Unreleased

### Explicit cross-site check

- Every `POST`, `PUT`, `PATCH` and `DELETE` now checks `Origin` and `Sec-Fetch-Site`. A request
  from a page on another site is refused with HTTP 403 and the new error code
  `CROSS_ORIGIN_REFUSED`. Before, such a write was refused only indirectly, because the API accepts
  JSON only and browsers preflight cross-origin JSON. Requests without these browser headers (the
  CLI, scripts, `curl`) are unaffected, and so are `GET`, `HEAD` and `OPTIONS`.
- The development server (`npm run dev`) presents proxied API requests with the backend's origin,
  so the guard accepts them.

## 0.1.0 — 17 November 2026

First public release of Venture Frame: the MCDA Studio engine, command line and local interface,
with four venture case studies in [`cases/`](cases/README.md).
