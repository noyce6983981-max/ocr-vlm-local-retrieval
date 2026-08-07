"""Protect lightweight CPU tests from optional model dependencies."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_cpu_environment_can_import_public_scripts() -> None:
    program = textwrap.dedent(
        """
        import importlib
        import importlib.abc
        import sys

        blocked = {
            "FlagEmbedding",
            "qwen_vl_utils",
            "torch",
            "transformers",
        }

        class BlockOptionalModels(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                del path, target
                if fullname.split(".", 1)[0] in blocked:
                    raise ModuleNotFoundError(
                        f"optional model dependency blocked: {fullname}"
                    )
                return None

        sys.meta_path.insert(0, BlockOptionalModels())
        for module in (
            "scripts.evaluate_text_retrieval",
            "scripts.index_user_text",
            "scripts.score_text_query",
            "scripts.train_learned_gate",
            "scripts.train_preference_gate",
        ):
            importlib.import_module(module)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
