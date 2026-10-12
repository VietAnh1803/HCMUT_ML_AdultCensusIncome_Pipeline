"""Run the discovered suite and record each executed case, without counting subtests twice."""
import argparse
import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cases = {}

    def startTest(self, test):
        super().startTest(test)
        self.cases[test.id()] = 'running'

    def addSuccess(self, test):
        super().addSuccess(test)
        if self.cases[test.id()] == 'running':
            self.cases[test.id()] = 'passed'

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.cases[test.id()] = 'failed'

    def addError(self, test, err):
        super().addError(test, err)
        self.cases[test.id()] = 'error'

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.cases[test.id()] = 'skipped'

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self.cases[test.id()] = 'failed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/verification/test_results.json')
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordedResult).run(suite)
    report = {'run_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
              'command': 'python tools/run_tests.py', 'tests_run': result.testsRun,
              'passed': sum(status == 'passed' for status in result.cases.values()),
              'failures': len(result.failures), 'errors': len(result.errors),
              'skipped': len(result.skipped), 'successful': result.wasSuccessful(),
              'cases': [{'id': name, 'status': status} for name, status in result.cases.items()]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f"Recorded {report['tests_run']} executed tests: {report['passed']} passed.")
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
