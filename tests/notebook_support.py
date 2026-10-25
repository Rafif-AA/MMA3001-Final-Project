"""Load the actual documented notebook code without running training cells."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / 'docs' / 'source-notebooks'
SAVED_TRIALS = list(range(1, 6))


def notebook_path(trial, saved=True):
    suffix = f'{trial}_5' if saved else str(trial)
    return SOURCES / f'MMA3001_Final_Project_Trial_{suffix}.ipynb'


def find_cell(trial, contains, saved=True):
    """Return the unique code cell containing a known function or operation."""
    path = notebook_path(trial, saved)
    notebook = json.loads(path.read_text(encoding='utf-8'))
    matches = [(number, ''.join(cell['source']))
               for number, cell in enumerate(notebook['cells'], 1)
               if cell['cell_type'] == 'code' and contains in ''.join(cell['source'])]
    if len(matches) != 1:
        raise ValueError(f'{path.name}: expected one cell containing {contains!r}, found {len(matches)}')
    return path, *matches[0]


def load_function(trial, name, namespace, saved=True):
    """Compile the unchanged function definition, including its nested helpers.

    Imports and top-level operations are intentionally not executed. Callers
    supply real numerical libraries or explicit external-service test doubles.
    """
    path, number, code = find_cell(trial, f'def {name}(', saved)
    tree = ast.parse(code)
    definitions = [node for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    if len(definitions) != 1:
        raise ValueError(f'Expected exactly one top-level definition of {name}')
    module = ast.Module(body=definitions, type_ignores=[])
    scope = dict(namespace)
    exec(compile(module, f'{path.name}:cell-{number}', 'exec'), scope)
    return scope[name]


def run_cell(trial, contains, namespace):
    """Execute one unchanged analysis cell against small controlled inputs."""
    path, number, code = find_cell(trial, contains)
    scope = dict(namespace)
    exec(compile(code, f'{path.name}:cell-{number}', 'exec'), scope)
    return scope

