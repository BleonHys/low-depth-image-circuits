import tempfile
import unittest
from pathlib import Path

from classifier.utils import vqc_training


class TestVQCPaths(unittest.TestCase):
    def test_data_dir_resolution(self):
        data_dir = Path("/tmp/fake_data_dir")
        config = {"data_dir": str(data_dir), "dataset_name": "dummy", "basepath": "/tmp/results"}
        resolved = vqc_training._resolve_data_path(config)
        self.assertEqual(resolved, str(data_dir / "dummy"))

    def test_missing_files_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "data_dir": tmpdir,
                "dataset_name": "missing_dataset",
                "basepath": "/tmp/results",
                "compression_depth": 0,
            }
            with self.assertRaises(ValueError) as ctx:
                vqc_training._load_dataset(config)
            self.assertIn("States file not found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
