"""Keep the shared Colab entry point parseable on every branch."""

import json
import io
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


class NotebookTests(unittest.TestCase):
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
             patch("builtins.input", side_effect=["1", "2", "1", "1", ""]) as ask, \
             redirect_stdout(io.StringIO()):
            exec(menu, namespace)
        self.assertEqual(namespace["selected_models"], ["bsrbf_kan"])
        self.assertEqual(namespace["selected_family"], "Neural")
        self.assertEqual(ask.call_count, 5)
