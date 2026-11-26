"""Django project with explicit access to sibling shared runtime packages."""
from pathlib import Path

__path__.append(str(Path(__file__).resolve().parents[1]))
