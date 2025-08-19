[pytest]

### adopts = -ra -q --strict-config --strict-markers --maxfail=1 --cov=.

- -ra: show extra summary (skipped/xfail reasons).

- -q: quieter output (clean CI logs).

- --strict-config: error on unknown config keys (catches typos).

- --strict-markers: only allow pre-declared markers (prevents marker typos/misuse).

- --maxfail=1: stop after first failure (fast feedback).

- --cov=.: enable coverage for the project (via pytest-cov).

- filterwarnings =

 - error: treat warnings as errors (forces clean runs).

 - ignore::DeprecationWarning:pkg_resources and ignore::DeprecationWarning:numpy:
silence known third-party deprecations so your build isn’t noisy for reasons outside your code.

- testpaths = tests

- restrict test discovery to the tests/ folder (predictable discovery).

- xfail_strict = true

 - turn XPASS into a failure (ensures xfail is not masking fixed tests).

[coverage:run]

- branch = True

 - measure branch coverage (if/else paths), not just line coverage.

- source = .

 - only measure your project’s code (excludes global site-packages).

- omit = */__init__.py, */site-packages/*

 - exclude boilerplate and third-party code from metrics.

[coverage:report]

fail_under = 85

fail CI if total coverage < 85% (quality gate).

skip_covered = True

hide files that are 100% covered (focus on gaps).

show_missing = True

list exact missing lines (actionable output).