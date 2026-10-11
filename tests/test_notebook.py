"""Keep the shared Colab entry point parseable on every branch."""

import json
import io
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


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

    def test_invalid_private_repo_token_offers_zip_upload(self):
        path = Path(__file__).resolve().parents[1] / "notebooks" / "Colab_Train_UJI.ipynb"
        source = "".join(json.loads(path.read_text(encoding="utf-8"))["cells"][1]["source"])
        memory = io.BytesIO()
        with zipfile.ZipFile(memory, "w") as archive:
            archive.writestr("repo/pyproject.toml", "[build-system]\n")

        google, colab = ModuleType("google"), ModuleType("google.colab")
        files, userdata = ModuleType("google.colab.files"), ModuleType("google.colab.userdata")
        files.upload = Mock(return_value={"repo.zip": memory.getvalue()})
        userdata.get = Mock(return_value="invalid-secret")
        colab.files, colab.userdata, google.colab = files, userdata, colab

        with tempfile.TemporaryDirectory() as temp:
            source = source.replace('root = Path("/content/repo_source")',
                                    f"root = Path({temp!r}) / 'download'")
            response = SimpleNamespace(ok=False, status_code=401)
            output = io.StringIO()
            with patch.dict(sys.modules, {"google": google, "google.colab": colab,
                                          "google.colab.files": files,
                                          "google.colab.userdata": userdata}), \
                 patch("requests.get", return_value=response), \
                 patch("subprocess.run") as commands, redirect_stdout(output):
                namespace = {}
                exec(compile(source, "shared bootstrap", "exec"), namespace)
            files.upload.assert_called_once()
            commands.assert_called_once()
            self.assertEqual(namespace["project"].name, "repo")
            self.assertIn("HTTP 401", output.getvalue())
            self.assertNotIn("invalid-secret", output.getvalue())
