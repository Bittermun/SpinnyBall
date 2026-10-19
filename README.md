# SpinnyBall

With Python 3.10 or later, from the repository root:

```sh
python -m workbench
```

Open http://127.0.0.1:8765/. Use --port to select another port or --no-browser to suppress automatically opening a browser.

Workbench checks:

```sh
python -m unittest discover -s workbench/tests -p "test_*.py"
```

The separate simulation package's dependencies are declared in pyproject.toml.
