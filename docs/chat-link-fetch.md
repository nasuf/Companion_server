# Public link preparation

Link metadata pages, Weibo/Toutiao helper APIs, Bilibili short links/APIs and
remote covers use `chat_links.fetch.public_link_client`. Search-provider
endpoints configured by the operator remain on their existing client; this
change does not redirect model, embedding or database traffic.

The transport uses the public HTTPX transport and HTTPcore network-backend
interfaces, with the already-qualified HTTPcore 1.0.9 dependency declared
explicitly. See [HTTPX transports](https://www.python-httpx.org/advanced/transports/)
and [HTTPcore network backends](https://www.encode.io/httpcore/network-backends/).
It does not patch library internals or replace the shared HTTP client.

## Network and resource boundary

- HTTP/HTTPS GET and HEAD only, no URL credentials or scoped IPv6 hosts.
- Resolve every new connection once and require **all** returned addresses to
  be public unicast. Loopback, private, link-local/metadata, shared/reserved,
  IPv4-mapped IPv6, 6to4, Teredo and well-known NAT64 ranges are rejected.
- Connect to the checked IP literal. Keep the original host for the Host
  header and certificate-checked TLS/SNI, using the same pinned certifi CA
  bundle as the existing HTTPX stack. Rebinding DNS cannot substitute a
  second hostname lookup before TCP connection.
- Validate the entire DNS answer before selecting at most eight distinct
  connection candidates. Both IPv6 and IPv4 retain a place when available;
  excess answers cannot create an unbounded number of tasks or sockets, and
  a private address beyond the candidate cap still rejects the destination.
- Redirect requests use the same transport, up to ten hops. Environment
  proxies and netrc are disabled for this dedicated client. There is no
  production switch for bypassing the address guard.
- Page bodies are capped at 6,000,000 decoded bytes before the existing
  1,500,000-character HTML parsing cap. Covers retain their 10 MiB cap.
  Wire bytes are separately capped at the decoded budget plus 64 KiB.
- Advertise gzip/deflate and decode incrementally with a bounded zlib output
  size, including raw deflate and concatenated gzip. Unsupported, corrupt or
  incomplete encodings fail explicitly instead of allocating an unbounded
  decoded chunk. HEAD response sizes are advisory; no response body is read.
- Existing connect/read timeouts apply; a shared 36-second wall-clock budget
  begins at the client's first request and covers all of its redirects and
  helper API requests. Streams and pools close on success, failure and timeout.

Ordinary public redirects, cookies and platform parsing remain supported.
A failed/blocked page keeps the existing partial-card/shared-text fallback;
a failed cover keeps the existing remote-image fallback. An internal-only
URL or a server that ignores the advertised encoding is no longer fetched by
this server. This is an intentional preparation boundary, not proof that
external provider pages always remain reachable or parseable.

This slice changes no schema, prompt, message/card identity, Web/Flutter wire
protocol, SQL rollout flag or checkpoint setting. Per-message link occurrence
and transactional SQL binding remain separate roadmap tasks.

## Verification and release

`test_chat_link_fetch.py` exercises the real HTTPcore/HTTPX protocol with
controlled DNS and network streams, plus a real isolated AnyIO/H11 socket
flow. It checks private/mixed DNS rejection, literal/numeric IPv4 and IPv6,
validated-IP dial and TLS host, DNS rebinding, every redirect, proxy isolation,
platform API/card fallback, cookies, read/decompression budgets, truncation,
stream cleanup and deadlines. The suite never probes production or providers.
`test_chat_links.py` retains platform/card/cache behavior coverage.

Release requires these tests, the full CI regression, chat protocol and
runtime role E2E, two reviews and staged hygiene. Production acceptance uses
serial standard-library/source and native read-only SQL probes as described
in `production-verification.md`; it must not import the application or run
untrusted URL/fault/load tests inside the serving API container.

## Immutable prepared versions and share history (R01.03.8.3.2.2)

The existing `chat_link_cards.id` remains the URL-cache identity. Each successful
preparation now commits an append-only `chat_link_snapshots` row in the same SQL
statement as the cache update. Snapshot JSON is capped at 128 KiB; its recorded
UTC time means preparation was stored, not that the website is factually true.
A failed snapshot insert rolls back the cache update too. An uncertain commit
acknowledgement never causes deletion of a cover known to be committed.

Cards carry an optional `payload.snapshot_id`. Sending it resolves that exact
version within the authenticated owner and conversation, without refetching the
page or trusting client-supplied body text. Missing/foreign versions produce a
correlated retry error. Clients that omit the key still resolve the existing
cache ID and retain the legacy refresh behavior. Flutter and the service-account
Web entry pass the card payload through unchanged; no required wire field changed.

Each persisted message already owns its `metadata.link_card` projection. Daily
share history now reads those message projections, so repeat shares of one URL
remain separate occurrences with their original content and timestamp. `id`
stays the cache ID; additive `occurrence_id` and existing `message_id` identify
the individual share. Reads return at most 1,000 items, or the requested smaller
limit; `total` is the number returned, not an all-time count. Legacy cache-only
rows are a fallback only where the bound message has no link projection; they
are not reconstructed or assigned a fabricated snapshot timestamp. Deleting a
message clears the cache's legacy pointer rather than cascading into versions
still used by other messages. Conversation/account deletion still cascades.

Web admin Agent conversation history displays the selected version, cache ID,
recorded UTC time and ready/partial/unrecorded state. It exposes no URL, body,
provider exception, query token or additional fetch request in this panel.
Detailed fetch failure categories and aggregate diagnostics remain O01.01.V01;
partial does not mean all content was read or no error occurred.

SQL ingress activation, execution-generation validation at atomic message
acceptance, quota/resource atomicity and durable recovery remain
R01.03.8.3.2.3/2.4. This slice does not enable checkpoints or the SQL link adapter.
Unused preview retention and total snapshot storage budgets remain an explicit
follow-up in that preparation/cleanup work; this batch performs no historical
purge or fabricated backfill. Cover cleanup on uncertain database availability
retains the file for later conversation/account cleanup.

Verification includes real PostgreSQL concurrency, rollback, lost commit ACK,
repeat sharing, legacy records, deleted-message retention and ownership tests;
real Chromium HTTP/JWT/history checks; existing Flutter daily-share tests and
payload round-trip checks; the production React admin overview at desktop sizes.
