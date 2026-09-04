import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / 'tools' / 'select_bridge_checkpoint.py'
SPEC = importlib.util.spec_from_file_location('select_bridge_checkpoint_standalone', MODULE_PATH)
SELECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECTOR)
METRICS = SELECTOR.METRICS
parse_metrics_json = SELECTOR.parse_metrics_json


class BridgeCheckpointSelectionTests(unittest.TestCase):
    def test_full_precision_json_is_preserved(self):
        expected = {metric: index + 0.123456 for index, metric in enumerate(METRICS)}
        report = {
            'results_percent': {
                'structural defects': expected,
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'metrics.json'
            path.write_text(json.dumps(report), encoding='utf-8')
            self.assertEqual(parse_metrics_json(path), expected)


if __name__ == '__main__':
    unittest.main()
