# Oxford Flowers 102 - Research Project

End-to-end image classification research workflow for Oxford Flowers 102 with a notebook-first setup.

## Project Structure

- `flower_data/flower_data/`: dataset root (`train/`, `valid/`, `test/`, `cat_to_name.json`)
- `notebooks/`: phase-by-phase notebooks
  - `01_data_audit.ipynb`
  - `02_baseline_train.ipynb`
  - `03_advanced_training.ipynb`
  - `04_hparam_search.ipynb`
  - `05_evaluation_error_analysis.ipynb`
  - `06_inference_demo.ipynb`
- `src/flowers102/`: reusable training/evaluation modules
- `configs/`: baseline/advanced/search configuration files
- `checkpoints/`: saved model checkpoints
- `reports/`: experiment outputs and final report

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

## Workflow by Phase

1. **Data audit**
   - Open and run `notebooks/01_data_audit.ipynb`.
   - Output: `reports/data_audit_report.json`.

2. **Baseline training**
   - Run `notebooks/02_baseline_train.ipynb`.
   - Output: `checkpoints/baseline_best.pth`, `reports/baseline_history.json`.

3. **Advanced training**
   - Run `notebooks/03_advanced_training.ipynb`.
   - Output: `checkpoints/advanced_stage1_best.pth`, `checkpoints/advanced_final_best.pth`.

4. **Hyperparameter search**
   - Run `notebooks/04_hparam_search.ipynb`.
   - Output: `reports/hparam_search_results.json`.

5. **Final evaluation and error analysis**
   - Run `notebooks/05_evaluation_error_analysis.ipynb`.
   - Output: `reports/final_evaluation.json`.

6. **Inference demo**
   - Run `notebooks/06_inference_demo.ipynb`.
   - Output: top-k predicted classes for a sample image.

## Optional CLI Baseline Run

```bash
python -m src.flowers102.experiment --mode baseline --data-dir flower_data/flower_data --out-dir .
```

## Recommended Targets

- Test Top-1: `>= 90%` (good), `>= 92%` (very good)
- Test Top-5: `>= 97%`

## Notes

- If GPU memory is limited, reduce `batch_size` in notebooks.
- For reproducibility, keep `seed=42` and avoid changing data splits during experiments.

