"""Check the Colab bootstrap without requiring a GitHub token or Colab runtime."""

import io
import json
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch
from urllib.error import HTTPError


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "Colab_Train_KNN.ipynb"


class KNNColabTests(unittest.TestCase):
    def test_code_cells_compile(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                compile("".join(cell["source"]), f"KNN cell {index}", "exec")

    def test_invalid_token_uses_uploaded_zip(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "".join(notebook["cells"][1]["source"])
        memory = io.BytesIO()
        with zipfile.ZipFile(memory, "w") as archive:
            archive.writestr("repo/pyproject.toml", "[build-system]\n")
            archive.writestr("repo/src/ips/neighbors/__init__.py", "")

        google, colab = ModuleType("google"), ModuleType("google.colab")
        files, userdata = ModuleType("google.colab.files"), ModuleType("google.colab.userdata")
        files.upload = Mock(return_value={"repo.zip": memory.getvalue()})
        userdata.get = Mock(return_value="invalid-secret")
        colab.files, colab.userdata, google.colab = files, userdata, colab

        with tempfile.TemporaryDirectory() as temp:
            source = source.replace('root = Path("/content/knn_repo_source")',
                                    f"root = Path({temp!r}) / 'download'")
            failure = HTTPError("https://api.github.com", 401, "Unauthorized", None, None)
            output = io.StringIO()
            with patch.dict(sys.modules, {"google": google, "google.colab": colab,
                                          "google.colab.files": files,
                                          "google.colab.userdata": userdata}), \
                 patch("urllib.request.urlopen", side_effect=failure), \
                 patch("subprocess.run") as commands, redirect_stdout(output):
                namespace = {}
                exec(compile(source, "KNN bootstrap", "exec"), namespace)
            files.upload.assert_called_once()
            self.assertEqual(namespace["project"].name, "repo")
            self.assertEqual(commands.call_count, 2)
            self.assertIn("HTTP 401", output.getvalue())
            self.assertNotIn("invalid-secret", output.getvalue())
