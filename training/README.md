# Colab training and Weights & Biases

Open [forecasting_colab.ipynb](forecasting_colab.ipynb) in Google Colab and select a GPU runtime. The notebook runs repository functions; it does not duplicate model or preprocessing code.

## Philippines regional dataset

Open [forecasting_regions_colab.ipynb](forecasting_regions_colab.ipynb) to run the same baselines and A/B/C workflow using the 16-region dataset. It clones the project into `/content`, uses Colab-local outputs, runs all three architectures in one session, and downloads a model ZIP after comparison. Add `GITHUB_TOKEN` and `WANDB_API_KEY` to Colab Secrets. The reviewed `dataset/philippines-regions-v1/` package is included on the training branch, so the notebook clone or update provides it automatically; no manual dataset copy is needed. The regional approved protocol inherits the international collection process; its region-specific diagnostic and identity mapping are recorded in [regional-data-review.md](../docs/regional-data-review.md). The model forecasts within-region Google Trends interest and does not rank regional index levels. The original notebook and three-country protocol remain unchanged.

## Setup

1. Place the project at MyDrive/Ceview_Transformer in your Google Drive, or change PROJECT_ROOT in the notebook.
2. Include model/, training/, the selected protocol JSON in experiments/, dataset/demand-v2-provisional/, the three source CSVs in data/, and artifacts/build_provisional_dataset.py. A Git clone alone does not include these ignored dataset/source files. The protocol lists every required file and checksum.
3. Add your W&B key to the repository-root [.env](../.env), under WANDB_API_KEY. Set WANDB_PROJECT and optionally WANDB_ENTITY for your account or team. [.env.example](../.env.example) contains the blank configuration for a new checkout.
4. Keep .env private and out of Git. The notebook reads it without printing the key. W&B config and artifacts use explicit fields/files, never the entire environment or project folder.
5. Keep outputs at the default MyDrive/CeviewTraining, or set OUTPUT_ROOT to another persistent Drive folder. Keep that location across reruns.
6. Set EXPERIMENT to "A" for your first session. Select "B" tomorrow and "C" in a later session, using the same project, protocol, output folder, and compatible runtime. Each selection runs only that experiment with its three seeds.
7. For the international notebook, complete the dataset review requirements below. For the regional notebook, use the approved training branch, which includes the regional dataset. Then select **Runtime > Run all**.

Dependencies install from [requirements.txt](requirements.txt). If packages were already imported before installation and preflight reports a mismatch, restart the runtime and run all cells again.

The notebook defaults to [forecasting-v2-wandb](../experiments/forecasting-v2-wandb.json). This explicitly revises the tracker from JSON/CSV to W&B while preserving the original dataset, architectures, seeds, metrics, and numerical settings. [forecasting-v1](../experiments/forecasting-v1.json) remains unchanged for historical reference. JSON and CSV are still saved to Drive as durable local records.

## What runs and appears

- Validation scores for last-value, seven-week moving-average, and 52-week seasonal-naive baselines.
- Only the selected experiment: A (BiLSTM) by default, B (Transformer), or C (BiLSTM + Transformer). Each selected experiment runs seeds 42, 123, and 2026; it does not automatically start the other experiments.
- Live training and validation MAE each epoch, early stopping, best-checkpoint restoration, and final training/validation metrics.
- Training curves, per-seed tables, parameter counts, timing, architecture means and sample standard deviations, and validation horizon/country/category/series breakdowns.
- Partial training/validation reports after A or B. Final comparison waits until all nine runs are saved across sessions; then validation-only selection uses the frozen seed 42 policy.
- One final test report for the selected artifact, including MAE, RMSE, sMAPE, MAPE coverage, forecast horizons, countries, categories, series, and four-/12-week demand-mean errors.
- Links to the W&B runs and comparison report.

Training curves report online batch losses during optimization. Final training metrics use the restored best checkpoint in evaluation mode. They need not equal the final epoch's training loss. Inference time measures one warmed full-validation pass, including batching and data transfer, divided by example count; GPU timing synchronizes before and after that pass.

If no neural architecture beats the best baseline, the notebook reports that outcome and does not evaluate the test partition. It never tests all candidates to choose a winner.

## Saved results

Under OUTPUT_ROOT/<protocol_version>/:

- suite.json: protocol identity, runtime, and code checksums.
- baselines.json: validation baseline reports.
- A-42/ and other architecture/seed folders: completed-run records and retained attempt folders containing best.pt, epochs.csv, and full train/validation metrics.
- selection.json: recorded validation decision.
- test.json: final test results, when a candidate qualifies.
- summary-synced.json: W&B comparison URL after successful synchronization.

W&B receives each epoch's metrics, final metric tables, configuration, hardware/software identity, and explicitly listed model/evaluation artifacts. Best checkpoints save to Drive whenever validation improves and upload to W&B on successful run completion. Full protocol metadata accompanies the weights. Raw dataset files, .env, and source folders are not W&B artifacts.

W&B runs use online mode. A logging or upload failure stops the workflow rather than claiming remote persistence. A completed local test report can be synchronized again without recomputing its metrics.

## Reruns and interrupted sessions

The EXPERIMENT selector changes session scheduling only; it does not change the frozen model settings or seed budget. Completed runs reload their checkpoint and results after checksum verification. Selecting A again reuses A; selecting B trains only missing B runs while preserving A. After C completes the remaining runs, comparison and final testing proceed automatically if a neural candidate qualifies. A runtime interruption during training starts a fresh attempt with the same seed on the next run; partial attempt files remain available. This is completed-run reuse, not optimizer-state resumption in the middle of an epoch.

Protocol, runtime, or code changes cannot silently reuse an existing suite. Keep a stable external environment across the comparison. Review a changed experiment rather than overwriting its records.

OUTPUT_ROOT/test-evaluations/<test-file-sha256>/ holds a persistent claim and final report. The claim is created before opening test targets. A rerun returns the existing report; a different selected artifact cannot claim that held-out set automatically. If the runtime stops after claiming test access but before saving the report, the workflow requires a ledger review and does not automatically retest. Do not delete or relocate this ledger to select candidates using test scores.

## Philippines regional protocol

The regional notebook uses `experiments/forecasting-regions-v2-approved.json`, a revision of the provisional regional protocol. It inherits the collection process and Colab/W&B use approved in `forecasting-v3-approved`, based on the data owner's confirmation that the regional exports differ only by geographic location. The regional diagnostic and preprocessing identity review are recorded in [regional-data-review.md](../docs/regional-data-review.md). Keep the `forecasting-regions-v1` protocol as historical record.

W&B documents [API-key environment variables](https://docs.wandb.ai/models/track/environment-variables) and [run initialization](https://docs.wandb.ai/models/ref/python/functions/init). See [Colab's FAQ](https://research.google.com/colaboratory/intl/en-GB/faq.html) for runtime and Drive behavior.
