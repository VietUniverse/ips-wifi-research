"""Keep the shared Colab entry point parseable on every branch."""

import json
import io
import subprocess
import sys
import tempfile
import unittest
import urllib.request
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch
from urllib.error import HTTPError


class NotebookTests(unittest.TestCase):
    def test_bad_token_stops_with_actionable_error(self):
        path = Path(__file__).resolve().parents[1] / "notebooks" / "Colab_Train_UJI.ipynb"
        source = "".join(json.loads(path.read_text(encoding="utf-8"))["cells"][1]["source"])
        google, colab = ModuleType("google"), ModuleType("google.colab")
        files, userdata = ModuleType("google.colab.files"), ModuleType("google.colab.userdata")
        files.upload = Mock()
        userdata.get = Mock(return_value="invalid-secret")
        colab.files, colab.userdata, google.colab = files, userdata, colab
        with tempfile.TemporaryDirectory() as temp:
            source = source.replace('root = Path("/content/repo_source")',
                                    f"root = Path({temp!r}) / 'download'")
            failure = HTTPError("https://api.github.com", 401, "Unauthorized", None, None)
            with patch.dict(sys.modules, {"google": google, "google.colab": colab,
                                          "google.colab.files": files,
                                          "google.colab.userdata": userdata}), \
                 patch("urllib.request.urlopen", side_effect=failure), \
                 redirect_stdout(io.StringIO()):
                namespace = {}
                with self.assertRaisesRegex(RuntimeError, "HTTP 401.*Contents: read"):
                    exec(compile(source, "shared bootstrap", "exec"), namespace)
            files.upload.assert_not_called()

    def test_bootstrap_scans_all_feature_branches_and_merges_new_plugins(self):
        path = Path(__file__).resolve().parents[1] / "notebooks" / "Colab_Train_UJI.ipynb"
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        setup = "".join(cells[1]["source"])
        self.assertIn('extract_branch("main", root)', setup)
        self.assertIn('startswith("feature/")', setup)
        self.assertIn('src", "ips", "algorithms"', setup)
        self.assertIn('"data", "preprocessors")', setup)
        self.assertIn('target.exists()', setup)

    def test_shared_colab_code_cells_compile_and_are_clean(self):
        path = Path(__file__).resolve().parents[1] / "notebooks" / "Colab_Train_UJI.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                compile("".join(cell["source"]), f"notebook cell {index}", "exec")

    def test_model_menu_waits_for_explicit_selection(self):
        path = Path(__file__).resolve().parents[1] / "notebooks" / "Colab_Train_UJI.ipynb"
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        menu = next("".join(c["source"]) for c in cells if c["cell_type"] == "code"
                    and "selected_family = choose_one" in "".join(c["source"]))
        namespace = {"project": path.parents[1], "Path": Path,
                     "subprocess": subprocess, "sys": sys}
        available = json.dumps({"models": ["mlp", "bsrbf_kan"],
                                "preprocessors": ["rss_standard"]})
        with patch.object(subprocess, "check_output", return_value=available), \
             patch("builtins.input", side_effect=["1", "1", "1", "1", ""]) as ask, \
             redirect_stdout(io.StringIO()):
            exec(menu, namespace)
        self.assertEqual(namespace["selected_models"], ["bsrbf_kan"])
        self.assertEqual(namespace["selected_family"], "Neural")
        self.assertEqual(ask.call_count, 5)

    def test_kan_mixer_is_an_explicit_menu_choice_with_mixer_presets(self):
        path = Path(__file__).resolve().parents[1] / "notebooks" / "Colab_Train_UJI.ipynb"
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        menu = next("".join(c["source"]) for c in cells if c["cell_type"] == "code"
                    and "selected_family = choose_one" in "".join(c["source"]))
        available = json.dumps({"models": ["bsrbf_kan", "mlp"],
                                "preprocessors": ["rss_standard"]})
        namespace = {"project": path.parents[1], "Path": Path,
                     "subprocess": subprocess, "sys": sys}
        # Display order: Neural, bsrbf_kan, KAN-Mixer, mlp, then RSS/preset/params.
        with patch.object(subprocess, "check_output", return_value=available), \
             patch("builtins.input", side_effect=["1", "2", "1", "1", ""]) as ask, \
             redirect_stdout(io.StringIO()) as output:
            exec(menu, namespace)
        self.assertEqual(namespace["selected_models"], ["kan_mixer"])
        self.assertTrue(json.loads((namespace["project"] / namespace["selected_preset"])
                                   .read_text())["with_mixer"])
        self.assertIn("KAN-Mixer (BSRBF-KAN + Residual Mixer)", output.getvalue())
        self.assertEqual(ask.call_count, 5)
