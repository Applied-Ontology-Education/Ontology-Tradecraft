"""Extra offline regression checks for Step 6 supervised timeouts.

No ROBOT/JVM/API needed; use a Python sleeper to validate that hanging
child commands fail instead of passing and logging is available.
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rdflib import URIRef
from rdflib.namespace import RDFS

import validate_rail_reasoner as v6
from test_validate_rail_reasoner import Step6Tests


class Step6WatchdogTests(unittest.TestCase):
    def test_robot_process_timeout_writes_log_and_returns_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_file = root / 'input.ttl'
            input_file.write_text('# fake OWL input', encoding='utf-8')
            output = root / 'never_written.ttl'
            log = root / 'robot.log'
            fake_robot = [sys.executable, '-c', 'import time; time.sleep(60)']
            ok, result = v6._reason(fake_robot, input_file, output, log, 'hermit', 1)
            self.assertFalse(ok)
            self.assertTrue(result['executed'])
            self.assertTrue(result['timed_out'])
            self.assertFalse(output.exists())
            self.assertIn('TIMEOUT: True', log.read_text(encoding='utf-8'))

    def test_prepare_only_does_not_run_reasoner(self):
        t = Step6Tests()
        t.setUp()
        try:
            with patch.object(v6, '_verify', return_value=t.context), \
                 patch.object(v6, '_robot_command', return_value=['robot']), \
                 patch.object(v6, '_reason', side_effect=AssertionError('ROBOT must not run')):
                result = v6.validate(t.root, prepare_only=True)
            self.assertEqual(result['status'], 'PASS_PREPARATION_ONLY_UNREASONED')
            self.assertFalse(result['owl_reasoner_executed'])
            self.assertTrue((t.root / 'reports/step6_prepare_summary.json').is_file())
            self.assertFalse((t.root / 'reports/step6_reasoner_summary.json').exists())
        finally:
            t.tearDown() if hasattr(t, 'tearDown') else None
            t.doCleanups()


if __name__ == '__main__':
    unittest.main()
