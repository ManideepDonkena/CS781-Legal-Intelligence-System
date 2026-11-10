"""Shared headless plotting and a rendering check before costly experiments."""
import copy
import importlib
import io
import sys


def validate_version(version, python_version=None):
    from packaging.version import Version
    python_version = sys.version_info if python_version is None else python_version
    if tuple(python_version[:2]) >= (3, 14) and Version(version) < Version("3.10.5"):
        raise RuntimeError(
            f"Matplotlib {version} has the Path.__deepcopy__ recursion bug on Python 3.14. "
            "Run: python -m pip install --upgrade --only-binary=:all: 'matplotlib==3.10.9'. "
            "Keep your prepared data and checkpoints; a plotting dependency upgrade alone does not change their identity.")


def pyplot():
    import matplotlib
    validate_version(matplotlib.__version__)
    matplotlib.use("Agg")
    return importlib.import_module("matplotlib.pyplot")


def probe():
    """Exercise Path deepcopy and actual PNG rendering; import alone misses this bug."""
    plt = pyplot()
    from matplotlib.path import Path
    path = Path.unit_rectangle()
    copied = copy.deepcopy(path)
    if copied is path or copied.readonly:
        raise RuntimeError("Matplotlib Path deepcopy failed to create a mutable independent copy")
    figure = None
    try:
        figure, axes = plt.subplots(figsize=(2, 1.5))
        axes.plot([0, 1], [1, 0], marker="o")
        figure.tight_layout()
        output = io.BytesIO()
        figure.savefig(output, format="png", dpi=60)
        data = output.getvalue()
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError("Matplotlib produced an invalid PNG")
        return {"path_deepcopy": "passed", "png_render": "passed", "png_bytes": len(data), "backend": "Agg"}
    finally:
        if figure is not None:
            plt.close(figure)


if __name__ == "__main__":
    import json
    print(json.dumps(probe(), indent=2))
