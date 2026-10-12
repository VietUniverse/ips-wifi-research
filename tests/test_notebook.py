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
    def test_bad_token_allows_zip_upload(self):
        path = Path(__file__).resolve().parents[1] / "notebooks" / "Colab_Train_UJI.ipynb"
        source = "".join(json.loads(path.read_text(encoding="utf-8"))["cells"][1]["source"])
        archive_bytes = io.BytesIO()
        with zipfile.ZipFile(archive_bytes, "w") as archive:
            archive.writestr("repo/pyproject.toml", "[build-system]\n")
        google, colab = ModuleType("google"), ModuleType("google.colab")
        files, userdata = ModuleType("google.colab.files"), ModuleType("google.colab.userdata")
        files.upload = Mock(return_value={"repo.zip": archive_bytes.getvalue()})
        userdata.get = Mock(return_value="invalid-secret")
        colab.files, colab.userdata, google.colab = files, userdata, colab
        with tempfile.TemporaryDirectory() as temp:
            source = source.replace('root = Path("/content/repo_source")',
                                    f"root = Path({temp!r}) / 'download'")
            failure = HTTPError("https://api.github.com", 401, "Unauthorized", None, None)
            output = io.StringIO()
            with patch.dict(sys.modules, {"google": google, "google.colab": colab,
                                          "google.colab.files": files,
                                          "google.colab.userdata": userdata}), \
                 patch("urllib.request.urlopen", side_effect=failure), \
                 patch("builtins.input", return_value="1"), \
                 patch("subprocess.run") as commands, redirect_stdout(output):
                namespace = {}
                exec(compile(source, "shared bootstrap", "exec"), namespace)
            files.upload.assert_called_once()
            commands.assert_called_once()
            self.assertEqual(namespace["project"].name, "repo")
            self.assertIn("HTTP 401", output.getvalue())
            self.assertNotIn("invalid-secret", output.getvalue())

    def test_branch_menu_uses_existing_feature_branches(self):
        path = Path(__file__).resolve().parents[1] / "notebooks" / "Colab_Train_UJI.ipynb"
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        setup = next("".join(c["source"]) for c in cells if c["cell_type"] == "code"
                     and "known_branches =" in "".join(c["source"]))
        selection = setup[setup.index("known_branches ="):setup.index("\nif token:\n", setup.index("Đang tải nhánh"))]
        response = io.BytesIO(json.dumps([{"name": "main"}, {"name": "feature/MMJ"},
                                          {"name": "feature/knn-wknn"}]).encode())
        namespace = {"token": "test-token", "REPO": "owner/repo", "urllib": urllib,
                     "json": json}
        with patch.object(urllib.request, "urlopen", return_value=response), \
             patch("builtins.input", return_value="2") as ask, \
             redirect_stdout(io.StringIO()):
            exec(selection, namespace)
        self.assertEqual(namespace["BRANCH"], "feature/MMJ")
        ask.assert_called_once()

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
