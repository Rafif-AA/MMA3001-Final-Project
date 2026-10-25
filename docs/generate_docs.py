"""Build offline HTML reference pages from the ten supplied Colab notebooks.

Uses only Python's standard library. Notebook cells are parsed, never executed.
Run: python docs/generate_docs.py
Refresh snapshots: python docs/generate_docs.py --source-dir PATH_TO_DOWNLOADS
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import html
import io
import json
import keyword
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SNAPSHOTS = ROOT / 'source-notebooks'
DATE = '25 October 2026'
FOLDER = 'https://drive.google.com/drive/folders/1L7CqHadYiNWEeMiAlu7-QRTLIzC0RylO'
REPO = 'https://github.com/Rafif-AA/MMA3001-Final-Project'

CSS = '''
:root{--ink:#183133;--muted:#52686a;--teal:#006c62;--line:#dce7e4;--paper:#fff;--soft:#f3f7f5;--mono:#edf4f1}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:28px}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.7 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:var(--teal);text-underline-offset:3px}a:hover{text-decoration-thickness:2px}h1,h2,h3,h4{line-height:1.2;letter-spacing:-.035em}h1{font-size:clamp(2.3rem,4.6vw,3.8rem);font-weight:700;max-width:900px;margin:12px 0 22px}h2{font-size:1.8rem;margin:48px 0 20px}h3{font-size:1.2rem;margin:24px 0 12px}p{margin:12px 0 20px}code,pre{font-family:Consolas,"Liberation Mono",monospace;font-size:.86em}code{background:var(--mono);padding:2px 5px;border-radius:4px;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f8f6;border:1px solid var(--line);padding:18px;border-radius:9px;line-height:1.6}pre code{padding:0;background:none;font-size:1em}.sidebar{position:fixed;inset:0 auto 0 0;width:270px;background:var(--soft);border-right:1px solid var(--line);padding:30px 22px;overflow:auto}.brand{display:block;font-size:20px;font-weight:750;text-decoration:none;letter-spacing:-.04em}.brand small{display:block;color:var(--muted);font-size:12px;letter-spacing:.12em;text-transform:uppercase;margin-top:3px}.navgroup{font-size:11px;text-transform:uppercase;letter-spacing:.14em;margin:24px 8px 8px;color:var(--muted);font-weight:700}.navlink{display:block;padding:7px 11px;margin:3px 0;border-radius:6px;color:var(--muted);text-decoration:none;font-size:14px;line-height:1.4}.navlink.active{background:#dcece5;color:#004d45;font-weight:700}.search{width:100%;font:inherit;font-size:13px;padding:10px 12px;border:1px solid #becfc9;border-radius:6px;background:white;margin-top:26px}.search:focus{outline:2px solid #238b7e;outline-offset:2px}.searchhint{font-size:11px;color:var(--muted);margin:5px 2px 20px}main{margin-left:270px;padding:48px clamp(24px,5vw,78px) 64px;max-width:1500px}.eyebrow{color:var(--teal);font-size:12px;text-transform:uppercase;letter-spacing:.15em;font-weight:750}.lead{font-size:19px;color:var(--muted);max-width:800px}.hero{padding:12px 0 30px;border-bottom:1px solid var(--line)}.pill{display:inline-block;background:#e3f1eb;color:#005c50;font-size:12px;font-weight:650;border-radius:99px;padding:4px 11px;margin:5px 5px 5px 0}.stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:28px 0}.stat{border:1px solid var(--line);border-radius:10px;padding:20px;background:var(--soft)}.stat strong{font-size:32px;display:block;line-height:1.2;color:var(--teal)}.stat span{font-size:13px;color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.card{border:1px solid var(--line);border-radius:10px;padding:22px}.card h3{margin:7px 0 10px}.card p{font-size:14px;color:var(--muted);margin:0 0 14px}.card a{font-weight:650}.meta{font-size:12px;color:var(--muted)}.notice{background:#f1f7f3;border-left:3px solid #258874;padding:15px 20px;margin:24px 0;font-size:14px}.tablewrap{overflow-x:auto}table{width:100%;border-collapse:collapse;margin:18px 0 28px;text-align:left;font-size:14px}th{background:var(--soft);font-size:12px;color:var(--muted);font-weight:700}th,td{padding:12px 14px;border-bottom:1px solid var(--line);vertical-align:top}td code{font-size:12px}td:first-child{font-weight:550}ul,ol{padding-left:24px}li{margin:7px 0}.toc{display:flex;flex-wrap:wrap;gap:8px 18px;margin:18px 0}.toc a{font-size:13px}.api{border:1px solid var(--line);border-radius:10px;padding:24px;margin:20px 0}.api h3{margin:0 0 10px;overflow-wrap:anywhere}.docstring{white-space:pre-wrap;color:var(--muted);font-size:14px}.small{font-size:13px;color:var(--muted)}.cell{border-top:1px solid var(--line);padding:24px 0}.cell h3{margin:0 0 12px}.cell summary{cursor:pointer;color:var(--teal);font-size:13px;font-weight:650}.cell details{margin-top:14px}.source{white-space:pre;overflow-x:auto;overflow-wrap:normal;padding:14px 0;counter-reset:line;max-height:620px;overflow-y:auto}.source .line{display:block;min-width:max-content;padding-right:20px;line-height:1.65}.source .line:before{content:attr(data-line);display:inline-block;text-align:right;width:40px;margin-right:20px;color:#778b83;user-select:none}.source .line:target{background:#dcefe5}.k{color:#77418d;font-weight:650}.s{color:#046b53}.c{color:#637c72;font-style:italic}.n{color:#a45114}.sig{background:#f1f6f3;border:0;padding:14px;font-size:13px}.empty{display:none}.footer{border-top:1px solid var(--line);margin-top:50px;padding-top:18px;font-size:12px;color:var(--muted)}.workflow{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.workflow div{border-top:3px solid var(--teal);padding:13px 0}.workflow b{display:block}.badge{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--teal)}button{font:inherit}[hidden]{display:none!important}a:focus-visible,summary:focus-visible{outline:2px solid var(--teal);outline-offset:3px}@media(max-width:950px){.sidebar{position:relative;width:auto;padding:18px 24px;border-right:0;border-bottom:1px solid var(--line)}.sidebar nav{display:flex;flex-wrap:wrap;gap:4px}.navgroup{width:100%;margin:10px 0 3px}.navlink{padding:5px 9px}.search{margin-top:15px;max-width:460px}.searchhint{margin-bottom:12px}main{margin-left:0;padding-top:28px}.stats{margin:20px 0}.hero{padding-top:0}}@media(max-width:600px){main{padding:24px 18px 40px}.grid,.workflow{grid-template-columns:1fr}.stats{gap:8px}.stat{padding:13px 10px}.stat strong{font-size:25px}.stat span{font-size:11px}.api{padding:16px}th,td{padding:10px}h2{font-size:1.5rem}}@media print{.sidebar,.search,.toc{display:none}main{margin:0;padding:10px}.source{max-height:none;font-size:10px}.card,.api{break-inside:avoid}a{color:inherit}}
'''

JS = '''
const search=document.querySelector('.search');
if(search)search.addEventListener('input',()=>{const q=search.value.trim().toLowerCase();document.querySelectorAll('[data-search]').forEach(el=>{el.hidden=!el.dataset.search.includes(q);});});
function reveal(){if(!location.hash)return;const el=document.getElementById(decodeURIComponent(location.hash.slice(1)));if(!el)return;for(let p=el.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;requestAnimationFrame(()=>el.scrollIntoView({block:'start'}));}
window.addEventListener('hashchange',reveal);reveal();
'''


def esc(value):
    return html.escape(str(value), quote=True)


def source(cell):
    value = cell.get('source', [])
    return ''.join(value) if isinstance(value, list) else value


def parse(code):
    # Preserve line numbering while allowing the Colab !pip/%pip syntax.
    clean = '\n'.join('# ' + line if line.lstrip().startswith(('!', '%')) else line for line in code.splitlines())
    return ast.parse(clean)


def functions(tree, prefix=''):
    result = []
    for child in ast.iter_child_nodes(tree):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = prefix + child.name
            result.append((name, child))
            result.extend(functions(child, name + '.'))
        elif not isinstance(child, (ast.ClassDef,)):
            result.extend(functions(child, prefix))
    return result


def code_html(code, cell_number):
    lines = code.splitlines()
    pieces = {i: [] for i in range(1, len(lines) + 1)}
    try:
        for token in tokenize.generate_tokens(io.StringIO(code).readline):
            kind = ('k' if token.type == tokenize.NAME and keyword.iskeyword(token.string) else
                    's' if token.type == tokenize.STRING else
                    'c' if token.type == tokenize.COMMENT else
                    'n' if token.type == tokenize.NUMBER else '')
            if not kind:
                continue
            for row in range(token.start[0], min(token.end[0], len(lines)) + 1):
                a = token.start[1] if row == token.start[0] else 0
                b = token.end[1] if row == token.end[0] else len(lines[row - 1])
                if b > a:
                    pieces[row].append((a, b, kind))
    except (tokenize.TokenError, IndentationError):
        pass
    output = []
    for number, line in enumerate(lines, 1):
        pos, chunks = 0, []
        for a, b, kind in sorted(pieces[number]):
            if a < pos:
                continue
            chunks += [esc(line[pos:a]), f'<span class="{kind}">{esc(line[a:b])}</span>']
            pos = b
        chunks.append(esc(line[pos:]))
        output.append(f'<span class="line" id="cell-{cell_number}-line-{number}" data-line="{number}">{"".join(chunks)}</span>')
    return '<pre class="source"><code>' + ''.join(output) + '</code></pre>'


def table(headers, rows):
    return '<div class="tablewrap"><table><thead><tr>' + ''.join('<th>' + h + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + str(c) + '</td>' for c in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def section_info(code):
    if not code.strip():
        return 'Empty cell', 'This cell contains no source code.', 'None.'
    if re.search(r'^[!%]pip ', code, re.M):
        return 'Install dependencies', 'Install Ultralytics into the current Colab runtime.', 'Changes the runtime package environment; internet access is required.'
    if 'def evaluate_and_plot' in code:
        return 'Evaluate the saved checkpoint', 'Mount Drive, load WEIGHTS_PATH, evaluate the validation split at the function defaults, and display F1, precision, recall and precision-recall curves. The returned model and metrics are used by later cells.', 'Reads checkpoint and dataset files; writes evaluation outputs to the actual metrics.save_dir and opens figures.'
    if 'normalized = np.divide' in code:
        return 'Render confusion matrices', 'Copy the raw confusion matrix, normalise each true-class column with safe division, then hide the wrinkle row and column for display. Normalising before filtering preserves the original denominators; retained columns need not sum to one.', 'Requires model and metrics from evaluation. Saves confusion_matrix_no_wrinkle.png and confusion_matrix_normalized_no_wrinkle.png.'
    if 'def collect_layer_activations' in code:
        return 'Profile the loaded model', 'Collect tensor shapes and activation counts from leaf modules for one dummy RGB image. Report parameter, module and feature/channel counts. Hook cleanup and restoration of module training modes run in a finally block.', 'Requires the evaluated model. Creates network, df and model-summary variables; no training is performed.'
    if 'metrics = model.val' in code:
        return 'Measure validation timing', 'Run validation with imgsz=1280, batch=1, device=0, rect=False, augment=False and plots=False. Read the per-image inference time and separately sum preprocessing, inference and postprocessing time.', 'Requires model, DATA_PATH and a CUDA GPU. Reassigns metrics; timing depends on the actual runtime.'
    if 'ridge_point' in code:
        return 'Estimate arithmetic intensity and roofline', 'Compute an FP32 memory-traffic approximation from parameters and twice the activation count, then divide FLOPs by bytes moved. Compare arithmetic intensity with the assumed hardware ridge point. Read the cell source for the exact constants and input size.', 'Produces an analytical estimate, not measured GPU memory traffic, achieved throughput, or training time.'
    if 'model.train(' in code:
        return 'Train the detector', 'Train the selected pretrained detector on the dataset YAML using the epoch limit and image size in this call. Inspect the run directory and best checkpoint produced by this execution.', 'Requires mounted Drive, dataset images/labels and suitable compute. Writes model weights, training metrics and plots.'
    if 'def make_hook' in code:
        return 'Inspect layer activations', 'Load the checkpoint, register hooks on leaf modules and run one 640 x 640 dummy image without gradients. Collect output shapes and per-image activation counts in a pandas table, then remove hooks after the forward pass.', 'Reads a hard-coded best.pt path and creates network, df and layer_info. Hook cleanup follows the forward pass in this original implementation.'
    if 'InteractiveSheet' in code:
        return 'Open the activation table in Sheets', 'Copy df, convert Output Shape entries to strings, and pass the table to google.colab.sheets.InteractiveSheet.', 'Requires the activation table and Google Colab; opens an interactive Sheets integration.'
    if 'total_layers' in code:
        return 'Summarise model statistics', 'Count model parameters, parameters requiring gradients, modules and output features/channels. Sum the previously recorded activations.', 'Requires network and df. Gradient flags describe current model state and do not prove which weights were trained.'
    if 'make_archive' in code:
        return 'Download run outputs', 'Create a ZIP archive of /content/runs and request its download through the Colab files interface.', 'Reads the runtime run directory; writes /content/runs.zip and triggers a browser download.'
    if 'shutil.copy' in code:
        return 'Save the trained checkpoint to Drive', 'Mount Drive, create the destination directory and copy the configured best.pt checkpoint when it exists.', 'Reads the source checkpoint and may replace the destination file with the same name.'
    if 'plot_files' in code or 'files_to_show' in code:
        return 'Display saved training plots', 'Load available result images from results_dir and display them using Matplotlib. Missing images are reported rather than silently substituted.', 'Requires previously generated plot files; displays figures without retraining.'
    if 'drive.mount' in code:
        return 'Mount Google Drive', 'Connect the runtime to Google Drive so that /content/drive paths can be accessed.', 'Interactive Google account authorisation may be required.'
    if 'YOLO(' in code:
        return 'Initialise the pretrained model', 'Create a YOLO object from the weight filename in this cell. In saved-weight notebooks, the evaluation cell subsequently replaces this object with the selected trial checkpoint.', 'May download pretrained weights. This cell alone does not train or evaluate the model.'
    return 'Notebook operation', 'See the exact source below for this operation.', 'Depends on the variables and files referenced in the cell.'


PARAMS = {
    'weights_path': ('str or Path', 'Path to the trained checkpoint to load.'),
    'data_path': ('str or Path', 'Dataset YAML containing paths and class-ID mapping.'),
    'project': ('str or Path', 'Parent directory for evaluation outputs.'),
    'image_size': ('int', 'Square dummy-image or evaluation resolution; see the signature for this notebook\'s default.'),
    'batch_size': ('int', 'Number of images in each validation batch.'),
    'network': ('torch.nn.Module', 'Loaded PyTorch network; device and dtype are taken from its parameters.'),
    'name': ('str', 'Named leaf module associated with the returned hook.'),
    'module': ('torch.nn.Module', 'Module whose forward pass invoked the hook.'),
    'inputs': ('tuple', 'Forward-hook inputs; accepted but not used by this hook.'),
    'output': ('object', 'Forward result. Only tensor outputs are counted.'),
}


def function_card(qualified, node, cell_number):
    signature = qualified + '(' + ast.unparse(node.args) + ')'
    doc = ast.get_docstring(node)
    descriptions = {
        'evaluate_and_plot': ('Load a saved checkpoint, run validation, render four curves and return the loaded model and its metrics.', 'tuple: (YOLO model, validation metrics). Also displays figures and saves evaluation outputs.'),
        'collect_layer_activations': ('Profile tensor outputs of leaf modules with one dummy image, without calculating gradients.', 'pandas.DataFrame with Layer, Type, Output Shape and Activations columns. Its attrs record image_size and profiled_model_id.'),
        'make_hook': ('Create a closure that records outputs for one named module.', 'A callable accepting (module, inputs, output). It appends records to the surrounding activation list.'),
        'hook': ('Record the output shape and activation count when the module output is a tensor.', 'None. Appends a record to the list captured by the closure; non-tensor outputs are skipped.'),
    }
    summary, returns = descriptions[node.name]
    parameters = [(f'<code>{esc(a.arg)}</code>', esc(PARAMS.get(a.arg, ('object', 'See source.'))[0]), esc(PARAMS.get(a.arg, ('object', 'See source.'))[1])) for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
    anchor = f'function-{cell_number}-{qualified.replace(".", "-")}'
    copied_doc = '<h4>Source docstring</h4><div class="docstring">' + esc(doc) + '</div>' if doc else '<p class="small">The source function has no docstring; the explanation above was added for this reference.</p>'
    return f'<article class="api" id="{anchor}"><span class="badge">Function reference · cell {cell_number}</span><h3>{esc(qualified)}</h3><pre class="sig">{esc(signature)}</pre><p>{esc(summary)}</p>' + table(['Parameter', 'Expected value', 'Purpose'], parameters) + '<h4>Returns and effects</h4><p>' + esc(returns) + '</p>' + copied_doc + f'<p><a href="#cell-{cell_number}-line-{node.lineno}">View definition at cell {cell_number}, line {node.lineno}</a></p></article>'


def metadata(row):
    trial = int(re.search(r'Trial (\d)', row['title']).group(1))
    saved = ',5' in row['title']
    notebook = json.loads((SNAPSHOTS / row['file']).read_text(encoding='utf-8'))
    cells = [(i, c, source(c)) for i, c in enumerate(notebook['cells'], 1)]
    refs = [(i, q, n) for i, c, s in cells if c['cell_type'] == 'code' for q, n in functions(parse(s))]
    imports = sorted({ast.unparse(n) for _, c, s in cells if c['cell_type'] == 'code' for n in ast.walk(parse(s)) if isinstance(n, (ast.Import, ast.ImportFrom))})
    train_size = 1280 if trial in (2, 5) else 640
    profile_size = 1280 if saved and trial in (2, 4) else 640
    return dict(row, trial=trial, saved=saved, notebook=notebook, cells=cells, refs=refs, imports=imports,
                slug=('weights' if saved else 'training') + f'-trial-{trial}', train_size=train_size, profile_size=profile_size)


def layout(title, body, items, current='index'):
    links = '<a class="navlink' + (' active' if current == 'index' else '') + '" href="index.html">Overview &amp; setup</a>'
    for saved, group in [(False, 'Training notebooks'), (True, 'Saved-weight notebooks')]:
        links += '<div class="navgroup">' + group + '</div>'
        for item in items:
            if item['saved'] != saved:
                continue
            search = ' '.join([item['title'], group] + [q for _, q, _ in item['refs']]).lower()
            links += f'<a class="navlink{" active" if current == item["slug"] else ""}" data-search="{esc(search)}" href="{item["slug"]}.html">Trial {item["trial"]}{",5" if saved else ""}</a>'
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="MMA3001 trial notebook source documentation, setup and function reference."><title>{esc(title)} | MMA3001 code reference</title><link rel="stylesheet" href="assets/style.css"><script src="assets/docs.js" defer></script></head>
<body><aside class="sidebar"><a class="brand" href="index.html">MMA3001<small>Code documentation</small></a><label for="search" class="small">Find a notebook or function</label><input class="search" id="search" type="search" placeholder="Search this reference…"><p class="searchhint">Try “weights”, “trial 2” or “activations”.</p><nav aria-label="Documentation">{links}</nav><p class="small">Offline HTML · {DATE}</p><a class="small" href="{REPO}">GitHub repository ↗</a></aside><main>{body}<footer class="footer">MMA3001 · Pork rasher packaging defect detection · Source snapshot: {DATE}. Generated by static analysis; notebook code was not executed. <a href="source-notebooks/manifest.json">Source manifest</a></footer></main></body></html>'''


def notebook_page(item, items):
    mode = 'Saved-weight evaluation' if item['saved'] else 'Training experiment'
    title = 'Trial ' + str(item['trial']) + (',5' if item['saved'] else '')
    purpose = 'Evaluate an existing trial checkpoint, display metrics and inspect the loaded model without retraining.' if item['saved'] else 'Train a YOLO26 detector, display training plots and inspect layer activations.'
    body = f'<header class="hero"><div class="eyebrow">{mode}</div><h1>{title}<br>Source reference</h1><p class="lead">{purpose}</p><span class="pill">{len([c for _, c, s in item["cells"] if c["cell_type"] == "code" and s.strip()])} code cells</span><span class="pill">{len(item["refs"])} function definitions</span><p class="small">Original title: {esc(item["title"])}</p><div class="toc"><a href="#configuration">Configuration</a><a href="#functions">Functions</a><a href="#cells">Cell-by-cell guide</a><a href="{item["url"]}">Open original in Colab ↗</a><a download href="source-notebooks/{item["file"]}">Download source snapshot</a></div></header>'
    body += '<h2 id="configuration">Configuration and dependencies</h2>'
    if item['saved']:
        constants = []
        for _, c, s in item['cells']:
            if c['cell_type'] != 'code':
                continue
            for node in parse(s).body:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in ('WEIGHTS_PATH', 'DATA_PATH', 'OUTPUT_PROJECT'):
                            constants.append((f'<code>{esc(target.id)}</code>', f'<code>{esc(node.value.value)}</code>'))
        constants += [('Validation defaults', '<code>split="val", image_size=1280, batch_size=4</code>'), ('Timing call', '<code>imgsz=1280, batch=1, device=0</code>'), ('Activation input', f'<code>{item["profile_size"]} x {item["profile_size"]}, batch=1</code>')]
        body += table(['Setting', 'Source value'], constants)
        body += '<div class="notice">The saved checkpoint determines the evaluated architecture. Cell 3 initially loads <code>yolo26n.pt</code>, but cell 4 replaces that object with <code>YOLO(WEIGHTS_PATH)</code>. The copied evaluation docstring and output-folder name say “Trial 2” in all five files; the checkpoint path identifies the intended trial.</div>'
        body += '<p>Run the evaluation cell before the confusion-matrix, activation and timing cells. Run the model-summary cell before roofline analysis. The final timing cell reassigns <code>metrics</code>, so rerun the plot-producing evaluation cell before regenerating its saved confusion matrices.</p>'
        if item['trial'] in (1, 3, 5):
            body += '<p class="small">This notebook uses fixed Nano-model constants for a 640 x 640 roofline estimate. They are source assumptions, not measurements of the current checkpoint. Validation and timing still run at 1280 x 1280.</p>'
        else:
            body += '<p class="small">This notebook profiles a separate FP32 copy at 1280 x 1280 for its FLOP estimate, checks for a positive finite result and reuses activation data only for the same model and image size.</p>'
    else:
        body += table(['Setting', 'Source value'], [('Pretrained model', '<code>yolo26' + ('n' if item['trial'] <= 3 else 's') + '.pt</code>'), ('Maximum epochs', str(200 if item['trial'] in (3, 5) else 100)), ('Training input', f'{item["train_size"]} x {item["train_size"]}'), ('Activation input', '640 x 640, batch=1'), ('Dataset YAML', '<code>/content/drive/MyDrive/DATASET 3 - Pork Rasher/data.yaml</code>')])
        body += '<p>Mount Drive before accessing the dataset. Run training before plot display, then point every checkpoint and plot path at that run. Analysis paths are literal strings in these files; inspect them before execution. Training outputs may use a numbered directory when a previous run already exists.</p>'
    body += '<details><summary>Imported modules in this notebook</summary><pre>' + esc('\n'.join(item['imports'])) + '</pre></details>'
    body += '<h2 id="functions">Function reference</h2><p>Signatures, defaults and source docstrings are extracted directly from the code. Parameter and return explanations provide additional reference context.</p>'
    body += ''.join(function_card(q, node, i) for i, q, node in item['refs'])
    body += '<h2 id="cells">Cell-by-cell guide</h2><p>Cell numbers include the original markdown cells. Expand a source panel to inspect the complete code with line numbers. Saved outputs and personal notebook metadata are excluded.</p>'
    for i, cell, code in item['cells']:
        if cell['cell_type'] != 'code':
            continue
        heading, purpose, effects = section_info(code)
        body += f'<section class="cell" id="cell-{i}"><span class="badge">Cell {i} · {len(code.splitlines())} lines</span><h3>{esc(heading)}</h3><p>{esc(purpose)}</p><p class="small"><b>Inputs / effects:</b> {esc(effects)}</p><details><summary>View cell {i} source</summary>{code_html(code, i)}</details></section>'
    return layout(title, body, items, item['slug'])


def overview(items):
    cell_count = sum(sum(c['cell_type'] == 'code' and bool(s.strip()) for _, c, s in item['cells']) for item in items)
    function_count = sum(len(item['refs']) for item in items)
    body = '<header class="hero"><div class="eyebrow">MMA3001 · Source documentation</div><h1>Pork rasher packaging<br>defect detection</h1><p class="lead">A guide to the training and saved-weight evaluation code for five YOLO26 experiments.</p><span class="pill">Google Colab</span><span class="pill">Ultralytics YOLO26</span><span class="pill">PyTorch</span><div class="stats"><div class="stat"><strong>10</strong><span>documented notebooks</span></div><div class="stat"><strong>' + str(cell_count) + '</strong><span>non-empty code cells</span></div><div class="stat"><strong>' + str(function_count) + '</strong><span>function definitions, including nested hooks</span></div></div><div class="toc"><a href="#start">Getting started</a><a href="#catalogue">Notebook catalogue</a><a href="#outputs">Outputs and calculations</a><a href="#sources">Source provenance</a></div></header>'
    body += '<div class="notice">These pages document the ten notebooks in the supplied Drive folder as of ' + DATE + '. They include exact source snapshots, function signatures, docstrings, dependencies and execution order. They do not claim that training or evaluation was rerun. Drive versions may differ from the earlier notebooks committed at the repository root.</div>'
    body += '<h2 id="start">Choose a workflow</h2><div class="workflow"><div><span class="badge">01 · Training</span><b>Start from pretrained weights</b>Use Trials 1–5 to run the training experiments. Provide the dataset and save the resulting best checkpoint.</div><div><span class="badge">02 · Saved weights</span><b>Reuse a trained checkpoint</b>Use Trials 1,5–5,5 to evaluate the corresponding saved weights. These notebooks contain no training call.</div><div><span class="badge">03 · Analysis</span><b>Inspect metrics and model structure</b>Display curves and matrices, profile activations and examine timing or theoretical roofline estimates.</div></div>'
    body += '<h3>Environment and required files</h3><ol><li>Open the selected original notebook in Google Colab. Choose a GPU runtime for training and for the saved-weight timing cells, which specify <code>device=0</code>.</li><li>Install the libraries used by the chosen notebook. The full set across the folder is shown below; Colab supplies its own integration modules.</li><li>Mount Drive before accessing <code>/content/drive</code>. Provide the dataset YAML, images and labels, and the trial checkpoint for saved-weight evaluation.</li><li>Update paths to match your files, then run cells in dependency order. Later analysis cells depend on <code>model</code>, <code>metrics</code>, <code>network</code> or <code>df</code> from earlier cells.</li></ol><pre>%pip install ultralytics torch matplotlib pandas pillow numpy seaborn ultralytics-thop</pre><p class="small">This is a Colab setup example. Package versions are not pinned in the supplied notebooks. <code>google.colab</code> modules are specific to that environment.</p><pre>from google.colab import drive\ndrive.mount("/content/drive")</pre>'
    body += '<h2 id="catalogue">Notebook catalogue</h2><p>Names retain the original comma suffix: “Trial 2,5” is the saved-weight notebook associated with Trial 2, not an extra model configuration.</p>'
    for saved, title in [(False, 'Training experiments'), (True, 'Saved-weight evaluation')]:
        body += '<h3>' + title + '</h3><div class="grid">'
        for item in items:
            if item['saved'] != saved:
                continue
            summary = f'Saved checkpoint for Trial {item["trial"]}; validation at 1280; activation profiling at {item["profile_size"]}.' if saved else f'YOLO26 {"Nano" if item["trial"] <= 3 else "Small"}; up to {200 if item["trial"] in (3, 5) else 100} epochs; {item["train_size"]} x {item["train_size"]} training input.'
            search = ' '.join([item['title'], title] + [q for _, q, _ in item['refs']]).lower()
            body += f'<article class="card" data-search="{esc(search)}"><span class="badge">{"Saved weights" if saved else "Training"}</span><h3>{esc(item["title"])}</h3><p>{esc(summary)}</p><a href="{item["slug"]}.html">Read source documentation →</a></article>'
        body += '</div>'
    body += '<h2 id="outputs">Outputs and calculations</h2>'
    body += table(['Operation', 'Inputs', 'Outputs / interpretation'], [
        ('Training', 'Pretrained weights, dataset YAML, epoch limit, image size', 'Run directory containing best.pt, last.pt, training metrics and plots.'),
        ('Saved-weight validation', 'WEIGHTS_PATH, DATA_PATH, validation settings', 'Loaded YOLO model, metrics and four performance curves saved under evaluation.save_dir.'),
        ('Confusion-matrix display', 'metrics.confusion_matrix and model.names', 'Raw and column-normalised matrices. Wrinkle is hidden after normalisation; background is retained.'),
        ('Activation profiling', 'Loaded network and dummy image resolution', 'Table of tensor output shapes and activation counts. Counts can include repeated leaf-module invocations.'),
        ('Timing', 'Validation at 1280, batch 1, CUDA device 0', 'Inference ms/image and the separately computed preprocessing + inference + postprocessing total.'),
        ('Roofline estimate', 'FLOPs, parameter count, activation count, reference peak/bandwidth', 'Arithmetic intensity and idealised compute ceiling. This is not a measurement of hardware throughput.'),
    ])
    body += '<h3>Roofline equations used in the newer notebooks</h3><pre>parameter_bytes = parameter_count * 4\nactivation_bytes = activation_count * 4 * 2\nestimated_bytes = parameter_bytes + activation_bytes\narithmetic_intensity = FLOPs / estimated_bytes\nridge_point = reference_peak_FLOPs_per_second / bandwidth_bytes_per_second\nceiling = min(reference_peak, arithmetic_intensity * bandwidth)</pre><p>Trials 2,5 and 4,5 estimate FLOPs for a 1280 x 1280 FP32 model copy. Trials 1,5, 3,5 and 5,5 retain fixed 640 x 640 Nano-model constants. The newer cells use a G4 reference of 117 TFLOP/s and 1597 GB/s as supplied in their source; these constants do not identify the GPU actually running the notebook.</p>'
    body += '<h2 id="sources">Source provenance and scope</h2><p>Source: <a href="' + FOLDER + '">MMA3001 Colab Files shared Drive folder</a>. Each notebook page links to the original Colab file and an offline source-only snapshot. Snapshots preserve every code and markdown cell; outputs, execution counts and personal metadata were removed. The manifest records original-file and snapshot SHA-256 hashes.</p><p>This reference was produced through static parsing with Python\'s standard library. Source docstrings are reproduced as written; accompanying explanations describe the actual calls and variables. Models and datasets were not downloaded or executed to build these pages.</p><p>To rebuild the documentation from the included snapshots:</p><pre>python docs/generate_docs.py</pre><p>The code is covered by the repository\'s <a href="../LICENSE">AGPL-3.0 licence</a>. Dataset: <a href="https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging">Pork Rasher Error(Packaging) by Hello</a>, listed under CC BY 4.0. The models have not yet been evaluated under real-world production-line conditions.</p>'
    return layout('Overview', body, items)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path)
    args = parser.parse_args()
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    if args.source_dir:
        manifest = json.loads((args.source_dir / 'manifest.json').read_text(encoding='utf-8'))
        for row in manifest:
            raw = (args.source_dir / row['file']).read_bytes()
            original = json.loads(raw)
            clean = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}}, 'cells': []}
            for i, cell in enumerate(original['cells'], 1):
                copy = {'cell_type': cell['cell_type'], 'id': 'cell-' + str(i), 'metadata': {}, 'source': cell.get('source', [])}
                if cell['cell_type'] == 'code':
                    copy.update(execution_count=None, outputs=[])
                clean['cells'].append(copy)
            payload = (json.dumps(clean, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
            (SNAPSHOTS / row['file']).write_bytes(payload)
            row['original_sha256'] = hashlib.sha256(raw).hexdigest()
            row['snapshot_sha256'] = hashlib.sha256(payload).hexdigest()
            row['snapshot_date'] = '2026-10-25'
        (SNAPSHOTS / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    manifest = json.loads((SNAPSHOTS / 'manifest.json').read_text(encoding='utf-8'))
    items = sorted([metadata(row) for row in manifest], key=lambda x: (x['saved'], x['trial']))
    (ROOT / 'assets').mkdir(exist_ok=True)
    (ROOT / 'assets/style.css').write_text(CSS.strip() + '\n', encoding='utf-8')
    (ROOT / 'assets/docs.js').write_text(JS.strip() + '\n', encoding='utf-8')
    (ROOT / 'index.html').write_text(overview(items), encoding='utf-8')
    for item in items:
        (ROOT / (item['slug'] + '.html')).write_text(notebook_page(item, items), encoding='utf-8')
    print(json.dumps({'notebooks': len(items), 'function_definitions': sum(len(x['refs']) for x in items), 'html_pages': len(items) + 1, 'index': str(ROOT / 'index.html')}))


if __name__ == '__main__':
    main()
