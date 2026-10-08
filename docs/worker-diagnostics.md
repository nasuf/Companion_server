# Worker failure snapshots

`python -m app.api_server` keeps the pinned Uvicorn0.54.0 two-worker
supervisor and60-second configured heartbeat decision. Healthy workers do
not receive a diagnostic signal and do not dump their stacks.

On Linux, the launcher creates a private0700 directory. CPython's spawn
bootstrap reimports the `-m` entry point as `__mp_main__`; before the application
imports, each child registers CPython's C-level `faulthandler` forSIGUSR2,
keeps a0600 dump descriptor open, and writes its registration. Registration
failure is optional and does not turn into an application startup failure.

When Uvicorn has already judged a worker unresponsive, the parent collects
small `/proc` and cgroup counters, verifies this session's parentPID,
workerPID, process start clock, and dump inode/device, then requests a trace
through a Linuxpidfd. It never falls back to sending a signal by numericPID.
An already exited, unregistered or mismatched process gets no diagnostic
signal. Platforms withoutpidfd support keep ordinary replacement behavior.
The caught-signal bit also has to remain present: a stale marker left after
unregistering the handler cannot cause a default-actionSIGUSR2. The launcher
reservesSIGUSR2 for this capture; application code must not replace its handler.

The additional collection wait has a200ms total monotonic deadline; OS I/O
and scheduling can overrun a wall-clock budget, so `capture_ms` reports actual
elapsed time. This is failure cleanup after the existing heartbeat decision,
not an increased heartbeat timeout. A stopped process is not resumed: its
native stopped state is recorded and its stack can be unavailable.

Only numeric resources, up to16 native thread states and32 Python frame
locations are logged. At most64KiB of a dump is read, favoring its tail and
current thread; larger dumps are marked truncated. Raw dumps, source lines,
locals, requests, credentials and exception messages are not logged.
CPython's own C dump limits100 threads/100 frames and500 characters per
string. There is one request per failed worker; the old sink and registration
are deleted afterjoin, and the session directory is removed on supervisor
shutdown. Normal HUP uses Uvicorn's ready-before-retire behavior and also
cleans retired sinks.

The snapshot shows observations, not a cause inferred from an exit code.
For example, `exitcode_after_join=-9` following a heartbeat miss indicates
the supervisor's subsequentSIGKILL; it does not prove an OOM. A regex frame
in a local synthetic GIL stall proves that the diagnostic can see that
controlled scenario; it does not reproduce a different production incident.

Qualification uses realCMD in owned network-isolated containers, including
healthy startup, GIL-held regex, stopped process, unregistered worker, startup
failure andHUP. Production acceptance is serial and read-only; do not
introduce a production fault or run app/ORM/graph/eval imports in the serving
container merely to test the diagnostic.

Public API references:

- [CPython3.13 faulthandler](https://docs.python.org/3.13/library/faulthandler.html)
- [CPython3.13 signals andpidfd](https://docs.python.org/3.13/library/signal.html#signal.pidfd_send_signal)
- [CPython3.13 spawn bootstrap](https://docs.python.org/3.13/library/multiprocessing.html#the-spawn-and-forkserver-start-methods)
