# Experiments

One folder per training run, written by `model_fine_tuning/*/train.py`:

```text
experiments/<name>/
  config.json          all arguments, model config, parameter counts
  log.jsonl            training / validation records (one JSON per line)
  val_metrics.json     validation metrics of the best checkpoint
  checkpoints/         best.pt, last.pt            (gitignored)
  eval_<split>/        evaluate.py output: metrics.json (+ per-image rows and overlays, gitignored)
```

`metrics.json` files and `config.json` are committed so results stay auditable without
the checkpoints. The log of what was run and what came out is in [LOG.md](LOG.md).
