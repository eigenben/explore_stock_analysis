# Explore Stock Analysis

Tiny sandbox for experimenting with stock-price forecasting using engineered technical features and a basic PyTorch MLP.

- **Data**: CSVs in `data/` (e.g., `nasdaq_19900101_20230630.csv`) with OHLCV columns and `Date` index.
- **Features**: Rolling means/std devs, volume ratios, and lagged returns built via `prepare_data_features` in `helpers.py`.
- **Model**: `NasdaqBasicMLPRunner` in `nasdaq_basic_mlp.py` trains a 2-layer MLP on standardized features and reports MSE/MAE/R².

## Quickstart (uv)
```
uv sync  # creates .venv and installs deps from uv.lock/pyproject.toml
uv run nasdaq_basic_mlp.py  # trains and prints test metrics
```

## Run on a remote GPU host
- Ensure rsync is installed: `apt update && apt install rsync -y`
- Install uv: `curl -sSL https://install.astral.sh | sh`
- Sync project to remote host: `bin/rspec 1.1.1.1:22 /workspace` (`/workspace` is implied/default)
- SSH into remote host and run: `cd /workspace && uv sync && uv run python nasdaq_basic_mlp.py`

## Notes
- Adjust train/test splits, learning rate, or epochs in the `NasdaqBasicMLPRunner` constructor.
- Swap `data/*.csv` or add new ones; filenames map to `load_stock_data("<name>")` without the `.csv` suffix.
