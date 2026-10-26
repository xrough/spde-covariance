"""Execute the illustrative notebook without installing Jupyter.

Usage: python scripts/check_notebook.py [notebook.ipynb] [--write-outputs]
Code cells use NumPy, Matplotlib, the installed research package, and a small
standard-library allowlist for reading committed evidence. Execution uses the
notebook's directory and restores the caller's directory even on failure. The
checker rejects magics and unexpected imports,
captures printed output and PNG figures, and checks every in-notebook assert.
The notebook is unchanged unless --write-outputs is explicitly supplied.
"""

from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import io
import json
import traceback
from pathlib import Path

_DEFAULT_NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "transport_covariance.ipynb"
_ALLOWED_IMPORTS = {"numpy", "matplotlib", "spde_covariance", "gzip", "hashlib", "json", "pathlib"}


def _source(cell):
    source = cell.get("source", "")
    if isinstance(source, list) and all(isinstance(line, str) for line in source):
        return "".join(source)
    if isinstance(source, str):
        return source
    raise ValueError("cell source must be a string or list of strings")


def _checked_ast(source, label):
    if any(line.lstrip().startswith(("%", "!")) for line in source.splitlines()):
        raise ValueError(f"{label}: IPython magics and shell commands are not supported")
    tree = ast.parse(source, filename=label, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module is None:
                raise ValueError(
                    f"{label}: relative imports create hidden working-directory requirements"
                )
            imports = [node.module]
        else:
            imports = []
        for module in imports:
            if module.split(".")[0] not in _ALLOWED_IMPORTS:
                raise ValueError(f"{label}: unexpected external requirement: {module}")
    return tree


def check_notebook(path: str | Path = _DEFAULT_NOTEBOOK, *, write_outputs: bool = False) -> dict:
    """Execute checked code cells sequentially; capture outputs without a kernel.

    This runs the trusted repository demonstration directly as Python. The
    static feature checks are reproducibility checks, not a sandbox for unknown
    notebooks. Relative data paths resolve from the notebook's directory; the
    caller's working directory is restored on success or failure. An explicitly
    requested output rewrite includes captured errors.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path = Path(path).resolve()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        raise ValueError("expected a notebook v4 document with a cells list")
    # Check all cells before running any of them.
    checked = []
    for index, cell in enumerate(notebook["cells"], start=1):
        if cell.get("cell_type") == "code":
            label = f"{path}:cell-{index}"
            checked.append((cell, _checked_ast(_source(cell), label), label))
    if not checked:
        raise ValueError("notebook must contain at least one code cell")

    namespace = {"__name__": "__notebook__"}
    current_outputs = []
    figures, output_count = 0, 0
    original_show = plt.show

    def capture_figures(*args, **kwargs):
        nonlocal figures
        for figure_number in plt.get_fignums():
            figure = plt.figure(figure_number)
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=100)
            current_outputs.append(
                {
                    "output_type": "display_data",
                    "metadata": {},
                    "data": {
                        "image/png": base64.b64encode(buffer.getvalue()).decode("ascii"),
                        "text/plain": f"<Figure {figure_number}: notebook verification>",
                    },
                }
            )
            figures += 1
            plt.close(figure)

    plt.show = capture_figures
    try:
        with contextlib.chdir(path.parent):
            for execution_count, (cell, tree, label) in enumerate(checked, start=1):
                current_outputs = []
                stdout, stderr = io.StringIO(), io.StringIO()
                cell["execution_count"], cell["outputs"] = execution_count, current_outputs
                error = None
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    try:
                        exec(compile(tree, label, "exec"), namespace)
                        capture_figures()
                    except Exception as exception:
                        error = exception
                streams = []
                for name, stream in (("stdout", stdout), ("stderr", stderr)):
                    if stream.getvalue():
                        streams.append(
                            {"output_type": "stream", "name": name, "text": stream.getvalue()}
                        )
                current_outputs[:0] = streams
                if error is not None:
                    current_outputs.append(
                        {
                            "output_type": "error",
                            "ename": type(error).__name__,
                            "evalue": str(error),
                            "traceback": traceback.format_exception(error),
                        }
                    )
                    raise RuntimeError(
                        f"notebook failed at execution cell {execution_count}: {label}"
                    ) from error
                output_count += len(current_outputs)
    finally:
        plt.show = original_show
        plt.close("all")
        if write_outputs:
            path.write_text(
                json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
    return {
        "notebook": str(path),
        "code_cells": len(checked),
        "captured_outputs": output_count,
        "figures": figures,
        "rewritten": write_outputs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", nargs="?", type=Path, default=_DEFAULT_NOTEBOOK)
    parser.add_argument(
        "--write-outputs",
        action="store_true",
        help="explicitly save execution counts and captured outputs into the notebook",
    )
    arguments = parser.parse_args()
    print(
        json.dumps(
            check_notebook(arguments.notebook, write_outputs=arguments.write_outputs), indent=2
        )
    )


if __name__ == "__main__":
    main()
