"""Fixtures and reporting for CPU unit tests of the documented notebook code."""
import hashlib
import html
import json
import platform
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest
import torch
from pytest_metadata.plugin import metadata_key

ROOT = Path(__file__).resolve().parents[1]
SCOPE = ('CPU software unit tests of the ten documented source snapshots. '
         'Real PyTorch, NumPy, pandas, Pillow, Matplotlib and Seaborn are used. '
         'YOLO checkpoint loading/validation and FLOP profiling are replaced by controlled test doubles. '
         'No model training, real checkpoint validation, accuracy evaluation, GPU timing, or production-line testing is performed.')


def fingerprints():
    manifest = json.loads((ROOT / 'docs/source-notebooks/manifest.json').read_text(encoding='utf-8'))
    sources = {row['file']: hashlib.sha256((ROOT / 'docs/source-notebooks' / row['file']).read_bytes()).hexdigest()
               for row in manifest}
    files = sorted((ROOT / 'tests').glob('*.py')) + [ROOT / 'pytest.ini', ROOT / 'requirements-test.txt']
    suite = {file.relative_to(ROOT).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest() for file in files}
    return sources, suite


def pytest_configure(config):
    torch.set_num_threads(1)
    config._mma_started = time.perf_counter()
    config._mma_started_at = datetime.now(ZoneInfo('Australia/Sydney')).isoformat()
    config._mma_versions = {name: version(name) for name in
                            ['pytest', 'pytest-html', 'torch', 'numpy', 'pandas', 'matplotlib', 'Pillow', 'seaborn']}
    config.stash[metadata_key] = {
        'Project': 'MMA3001 - Pork Rasher Packaging Defect Detection',
        'Scope': SCOPE,
        'Started (Australia/Sydney)': config._mma_started_at,
        'Python': platform.python_version(),
        'Platform': platform.platform(),
        'Device used': 'CPU',
        'Libraries': config._mma_versions,
        'Documented source snapshots': 10,
    }


@pytest.hookimpl(optionalhook=True)
def pytest_html_report_title(report):
    report.title = 'MMA3001 - Software Unit Test Report'


@pytest.hookimpl(optionalhook=True)
def pytest_html_results_summary(prefix, summary, postfix):
    prefix.extend(['<h2>Test scope</h2><p>' + html.escape(SCOPE) + '</p>',
                   '<p>Trial numbers in brackets identify parameterised cases against each notebook implementation. '
                   'A passed case is evidence of the stated software behaviour, not detector accuracy.</p>'])


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    result = yield
    report = result.get_result()
    report.description = (item.function.__doc__ or '').strip()


@pytest.hookimpl(optionalhook=True)
def pytest_html_results_table_header(cells):
    cells.insert(2, '<th>Verified behaviour</th>')


@pytest.hookimpl(optionalhook=True)
def pytest_html_results_table_row(report, cells):
    cells.insert(2, '<td>' + html.escape(getattr(report, 'description', '')) + '</td>')


def pytest_sessionfinish(session, exitstatus):
    reporter = session.config.pluginmanager.get_plugin('terminalreporter')
    sources, suite = fingerprints()
    counts = {key: len(reporter.stats.get(key, [])) for key in
              ['passed', 'failed', 'error', 'skipped', 'xfailed', 'xpassed']}
    records = [{'id': report.nodeid, 'outcome': outcome, 'duration_seconds': report.duration,
                'description': getattr(report, 'description', '')}
               for outcome in counts for report in reporter.stats.get(outcome, [])]
    summary = {
        'project': 'MMA3001', 'scope': SCOPE,
        'started_at': session.config._mma_started_at,
        'finished_at': datetime.now(ZoneInfo('Australia/Sydney')).isoformat(),
        'timezone': 'Australia/Sydney', 'exit_code': int(exitstatus),
        'collected': session.testscollected, 'results': counts,
        'duration_seconds': time.perf_counter() - session.config._mma_started,
        'python': platform.python_version(), 'platform': platform.platform(),
        'libraries': session.config._mma_versions,
        'source_snapshot_sha256': sources, 'test_suite_sha256': suite,
        'cases': sorted(records, key=lambda record: record['id']),
    }
    report_dir = ROOT / 'reports'
    report_dir.mkdir(exist_ok=True)
    (report_dir / 'test_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')

