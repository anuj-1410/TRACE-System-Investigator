# TRACE

TRACE is a local Streamlit prototype for reviewing structured system activity. It orders events, estimates an incident category, shows evidence and affected entities, and produces a report with suggested checks. It does not take response actions.

## Setup

Use **64-bit Python 3.12**. The project uses TensorFlow 2.21, scikit-learn, and Streamlit.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On macOS or Linux, create the environment with `python3.12 -m venv .venv`, activate it, then run the same install and Streamlit commands.

Select a built-in case, or upload one of the [seven separate demo CSVs](data/showcase/README.md). The headline percentages are model scores, not confirmed probabilities of a breach. Analysis runs locally.

## Training data

[The dataset card](data/DATASET_CARD.md) gives exact counts and limitations. The current seeded training set has **2,985 cases and 34,831 simulated events** across seven uneven classes. It has variable case lengths, benign context, overlapping event patterns, and varied users, hosts, timing, severity, processes, destinations, and transfer sizes.

The training set is **synthetic**, despite its realistic log fields. It is not captured enterprise data. A test split from the same generator cannot establish real-world accuracy. [LANL enterprise telemetry](https://csr.lanl.gov/data/2017/), [OTRF recorded lab captures](https://github.com/OTRF/Security-Datasets), and [CICIDS2017 flows](https://www.unb.ca/cic/datasets/ids-2017.html) use different schemas or labels; none is silently relabeled as all seven TRACE categories.

## Train and evaluate

The included model run can be reproduced with:

```powershell
python scripts/train_models.py --vae-epochs 20 --rbm-epochs 25 --batch-size 256
python scripts/evaluate_models.py
```

Training regenerates `data/trace_synthetic_cases.csv`, splits by `incident_id` into **60% fit, 20% normal-case calibration, and 20% untouched test**, then fits the TensorFlow VAE, contrastive-divergence RBM, multiclass classifier, and Markov timeline model. It saves model files and numeric metrics in `artifacts/`. Training takes several minutes on a CPU.

| Seed 41 result on 597 simulated test cases | Score |
| --- | ---: |
| Classifier macro F1 (fit / test) | 0.783 / 0.777 |
| Classifier balanced accuracy | 0.787 |
| VAE incident anomaly F1 / ROC AUC | 0.872 / 0.938 |
| Full pipeline category macro F1 / accuracy | 0.775 / 0.809 |
| Full pipeline incident false-positive rate | 11.0% |

These are **same-generator simulation results**. The separate recorded-lab check below exposed category errors and limited event mapping.

For your own reviewed case labels, use `python scripts/train_models.py --labelled-csv path/to/cases.csv`. Use `--normal-csv path/to/normal.csv` only for separately curated normal VAE training data. Keep all events from an incident in one partition. For a credible field estimate, also hold out a different organization or time period.

| Saved file | Purpose |
| --- | --- |
| `vae_encoder.keras`, `vae_decoder.keras`, `vae_metadata.json` | Native Keras VAE models and threshold |
| `rbm.keras`, `rbm_metadata.json` | Native Keras reconstruction graph and RBM parameters |
| `incident_classifier.joblib` | Fitted scikit-learn classifier; load only trusted local artifacts |
| `markov.json` | Transition probabilities |
| `training_summary.json` | Training settings, curves, and classifier/VAE/RBM/Markov metrics; also shown in the dashboard |
| `evaluation_audit.json` | Full-pipeline test results and errors; also shown in the dashboard |
| `split_manifest.json` | Case IDs for each partition |
| `recorded_telemetry_check.json` | Optional recorded-lab evaluation shown in the notebook |

### Recorded telemetry check

```powershell
python scripts/evaluate_recorded_telemetry.py
```

This downloads two small, public [OTRF attack-emulation captures](https://github.com/OTRF/Security-Datasets), maps supported Windows event IDs, and saves `artifacts/recorded_telemetry_check.json`. It requires internet access. The captures were recorded on lab systems and cover only **two** of the seven categories. Their results are a domain-shift check, not an overall accuracy figure.

In the included run, only **17/437** and **146/2,067** captured events mapped into the current event vocabulary. Both source tactics were misclassified. TRACE detected concerning activity in each, but this check does not support a real-world category-accuracy claim.

### Saved notebook results

```powershell
python -m pip install -r requirements-notebook.txt
```

Open [the training and evaluation notebook](notebooks/trace_training_evaluation.ipynb) in VS Code with its Jupyter extension, or in an existing Jupyter installation. Select this project's Python environment, run all cells, and save the notebook to retain updated tables and plots. It shows class counts, split checks, artifact checksums, training curves, per-class precision/recall/F1, confusion matrices, VAE anomaly metrics, RBM reconstruction, Markov transitions, end-to-end results, and the separate showcase and recorded-capture checks. `RUN_TRAINING = False` uses the saved run; set it to `True` to retrain when running all cells.

## Input CSV

The required fields are `timestamp`, `event_type`, and, for training, `incident_id` and one of the seven supported `incident_type` labels. Recommended fields are `user`, `system`, `source`, `destination`, `process`, `resource`, `severity`, `status`, `bytes_transferred`, and `raw_event`. See [the showcase CSVs](data/showcase/README.md) for examples. A dashboard upload can omit labels; training cannot.

## Limits

TRACE is an academic investigation aid. Synthetic test scores can be optimistic even with separate case IDs. The Bayesian root-cause probabilities and response-effect estimates are specified assumptions, not learned from the training set. Review findings against original logs and analyst-confirmed cases before relying on them.
