# Generated Prisma client memory

Use `recursive_type_depth = -1` in the Python client generator. This shares
recursive relation types instead of expanding pseudo-recursive types to depth
five. Nested reads and writes remain supported; model fields, datasource,
migrations and query-engine versions are unchanged. This option is supported by
[Prisma Client Python](https://prisma-client-py.readthedocs.io/en/stable/reference/config/#recursive-type-depth).
Recursive type checking requires a checker that supports recursive types, such
as Pyright; this setting does not promise compatibility with mypy's limitations.

The old schema generated 111,091 type classes in a 35.6 MB `types.py`. In the
qualified CPython 3.13 image, a fresh import consumed about 1.4 GiB RSS per
process. The two API workers therefore imposed substantial memory pressure
before handling a large nightly memory query. A naturally failed worker was
captured waiting for a memory page with about 779 MiB swapped out, while parsing
the nightly AiMemory query. No OOM was recorded. This establishes a resource
problem; it does not retrospectively prove the cause of every historical exit.

Qualification must include a fresh-interpreter RSS check, real PostgreSQL
create/update/serialization and a relation chain deeper than five, plus the
normal runtime image and two-worker tests. The resource probe creates synthetic
records only. Production acceptance uses numerical process/cgroup counters and
natural worker diagnostics; never import app/Prisma/graph modules in an extra
serving-container probe or force a production failure.

This is a client-generation change, with no database DDL. Compare the datamodel
before and after using `prisma migrate diff`, and verify migration hashes remain
unchanged. Do not create an empty migration merely for a generator option.
