# Test report

The recorded CPU software-test run on **25 October 2026 (Australia/Sydney)** completed with **78 passed, 0 failed, 0 errors and 0 skipped**.

- [HTML report](test_report.html): self-contained pytest report with each case, its verified behaviour and actual outcome. Download the file and open it in a browser; GitHub displays its HTML source.
- [Machine-readable summary](test_summary.json): outcomes, durations, exact library versions, source-snapshot hashes and test-suite hashes.
- [JUnit XML](junit.xml): machine-readable results for other test-reporting tools.
- [Tests and rerun instructions](../tests/README.md).

The run checks activation profiling, forward hooks, evaluation orchestration, matrix normalisation, plot generation, timing arithmetic and roofline calculations in the ten documented notebook snapshots. It uses real numerical/plotting libraries, while checkpoint loading, detector validation and FLOP profiling are controlled test doubles.

Passing these unit tests does not certify model accuracy, GPU timing or real-world production readiness. No full training run, real checkpoint validation or production-line evaluation was performed.
