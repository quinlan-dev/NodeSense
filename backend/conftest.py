"""Makes backend/ importable from tests regardless of pytest's invocation
directory or import mode — app.py, data.py, train.py etc. use bare
`from data import ...` style imports that assume backend/ is on sys.path."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
