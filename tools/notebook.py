"""Build the same reproducible walkthrough for local Python and Google Colab."""
import nbformat as nbf


def create_notebook():
    from pathlib import Path
    requirements = (Path(__file__).resolve().parents[1] / 'requirements-improved.txt').read_text(encoding='utf-8')
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md('# Adult Income: Classical ML, PyTorch and scoring\n\n'
           'Use **Runtime → Run all** on Google Colab with CPU. Setup runs before data-library imports. '
           'No Drive mount, Kaggle token or GPU is required. Internet is needed for public source, dependencies and UCI data.\n\n'
           'The default walkthrough reads the recorded experiment and runs tests plus the public-validation scorer. '
           'Set `REUSE_RECORDED_RUN = False` to train again. Previously observed test results are not an untouched holdout.'),
        code("from pathlib import Path\nimport importlib.util\nimport json\nimport subprocess\nimport sys\n\n"
             "print('Notebook Python:', sys.version, flush=True)\n"
             "ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'src/adult_benchmark.py').is_file()), None)\n"
             "if ROOT is None:\n"
             "    ROOT = Path.cwd() / 'adult-income-project'\n"
             "    if not (ROOT / 'src/adult_benchmark.py').is_file():\n"
             "        if ROOT.exists():\n"
             "            raise RuntimeError('adult-income-project exists without the source; choose an empty working directory.')\n"
             "        subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/VietAnh1803/HCMUT_ML_AdultCensusIncome_Pipeline.git', str(ROOT)], check=True)\n"
             "IN_COLAB = importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') is not None else False\n"
             "INSTALL_DEPENDENCIES = IN_COLAB  # Set True locally only if dependencies are missing.\n"
             f"NOTEBOOK_REQUIREMENTS = {requirements!r}\n"
             "if INSTALL_DEPENDENCIES:\n"
             "    if sys.version_info[:2] not in [(3, 12), (3, 13)]:\n"
             "        raise RuntimeError('Use a Python 3.12 or 3.13 runtime; other versions have not been validated.')\n"
             "    requirements_path = ROOT / '.cache/notebook-requirements.txt'\n"
             "    requirements_path.parent.mkdir(parents=True, exist_ok=True)\n"
             "    requirements_path.write_text(NOTEBOOK_REQUIREMENTS, encoding='utf-8')\n"
             "    print('Installing CPU PyTorch...', flush=True)\n"
             "    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--only-binary=:all:', 'torch==2.9.1', '--index-url', 'https://download.pytorch.org/whl/cpu'], check=True, timeout=600)\n"
             "    print('Installing pinned notebook dependencies...', flush=True)\n"
             "    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--only-binary=:all:', '-r', str(requirements_path)], check=True, timeout=600)\n"
             "from IPython.display import display, Image\nimport pandas as pd\n"
             "def run_script(relative, *arguments):\n"
             "    subprocess.run([sys.executable, str(ROOT / relative), *map(str, arguments)], cwd=ROOT, check=True)\n"
             "print('Source:', ROOT)\nprint('Colab:', IN_COLAB)\n"
             "import importlib.metadata\n"
             "print('Installed packages:', {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'torch', 'lightgbm']})"),
        md('## Recorded results or fresh training\n'
           'The default displays measured, shipped scores without retraining. Fresh training uses the same '
           'configurations in a new Python process, so package versions do not depend on already-imported kernel modules.'),
        code("REUSE_RECORDED_RUN = True\nOUT = ROOT / 'reports/improved'\n"
             "if not REUSE_RECORDED_RUN or not (OUT / 'metrics.json').is_file():\n"
             "    OUT = ROOT / '.cache/notebook-retrain'\n"
             "    run_script('src/adult_benchmark.py', '--output', OUT, '--cache', ROOT / '.cache/adult')\n"
             "else:\n    print('Recorded CPU experiment: this cell does not retrain.')\n"
             "report = json.loads((OUT / 'metrics.json').read_text())\n"
             "display(pd.Series(report['split']).to_frame('Split evidence'))"),
        md('## Run all 50 automated cases\n'
           'These verify implementation behavior, including scoring, leakage checks, early stopping and actual checkpoint weights.'),
        code("TEST_RECORD = ROOT / '.cache/notebook-test-results.json'\n"
             "run_script('tools/run_tests.py', '--output', TEST_RECORD)\n"
             "tests = json.loads(TEST_RECORD.read_text())\n"
             "assert tests['successful'] and tests['tests_run'] == tests['passed'] == 50\n"
             "print(f\"AUTOMATED_TESTS_COMPLETE: {tests['passed']}/{tests['tests_run']} passed\")"),
        md('## Public-validation scoring demo\n'
           'The logistic reference fits on training only and reads no evaluation labels. The scorer joins by ID. '
           'The 6,508-row validation labels are public and already used for selection; this is not a hidden challenge or a new holdout.'),
        code("DEMO = ROOT / '.cache/notebook-scoring'\n"
             "run_script('src/local_evaluation.py', 'prepare', '--output', DEMO)\n"
             "run_script('src/local_evaluation.py', 'reference', '--train', DEMO / 'train.csv', '--features', DEMO / 'evaluation_features.csv', '--output', DEMO / 'predictions.csv')\n"
             "run_script('src/local_evaluation.py', 'score', '--labels', DEMO / 'evaluation_labels.csv', '--predictions', DEMO / 'predictions.csv', '--output', DEMO / 'score.json')\n"
             "score = json.loads((DEMO / 'score.json').read_text())\n"
             "assert score['n'] == 6508\n"
             "display(pd.Series(score).to_frame('Fresh public-validation score'))\n"
             "print('PUBLIC_SCORING_COMPLETE')"),
        md('## Validation comparison\nModel and threshold selection use validation, excluding the predefined controls.'),
        code("validation = pd.json_normalize(report['validation_experiments'])\n"
             "display(validation[['name', 'family', 'validation.average_precision', 'validation.roc_auc']].sort_values('validation.average_precision', ascending=False))\n"
             "display(pd.Series(report['selection']).to_frame('Frozen choices'))"),
        md('## Filtered official test and three MLP seeds\n'
           'These are the experiment selected above: recorded by default, freshly trained only when reuse is disabled. '
           'Scores refer to a previously observed public test, not independent confirmation.'),
        code("display(pd.DataFrame(report['test']).T[['accuracy', 'average_precision', 'roc_auc', 'f1_positive']])\n"
             "display(pd.DataFrame(report['selected_mlp_seed_repeats'])[['seed', 'accuracy', 'average_precision', 'roc_auc']])\n"
             "assert all(report['checks'].values())\n"
             "print('ADULT_REPORT_VERIFIED: recorded/fresh protocol checks passed')"),
        md('## Actual experiment plot\nThe MLP remains behind LightGBM. Accuracy differences also reflect validation-selected thresholds. '
           'One split and three seeds do not establish generalization to current populations.'),
        code("display(Image(filename=str(OUT / 'comparison.png')))\n"
             "display(pd.Series(report['environment']).to_frame('Experiment environment'))"),
    ]
    nb.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nb.metadata.language_info = {'name': 'python'}
    nb.metadata.colab = {'name': 'adult_income_improved.ipynb', 'provenance': []}
    # nbformat otherwise assigns random IDs, making identical builds differ.
    for index, cell in enumerate(nb.cells):
        cell.id = f'adult-{index:02d}'
    nbf.validate(nb)
    return nb


if __name__ == '__main__':
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    nbf.write(create_notebook(), root / 'notebooks/adult_income_improved.ipynb')
    print('Generated seven-code-cell Colab/local walkthrough.')
