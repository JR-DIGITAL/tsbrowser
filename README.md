# tsbrowser

`tsbrowser` is an interactive tool for browsing, interpreting, and labelling Sentinel-2 time series. Full documentation is available at https://jr-digital.github.io/tsbrowser/.

## Environment setup with `uv`

[`uv`](https://docs.astral.sh/uv/) handles the virtual environment, Python interpreter, and dependency installation described in `pyproject.toml`.

```bash
uv venv --python 3.11
source .venv/bin/activate
# create .venv/ and install tsbrowser + runtime deps (uses uv.lock if present)
uv sync

# confirm the CLI is available
tsbrowser --help
```

- Use `uv sync --group dev` if you also need the documentation tooling defined in the `dev` dependency group.
- You can avoid activating the environment by prefacing commands with `uv run`, e.g. `uv run tsbrowser --help`.

## Using the `tsbrowser_demo_setup` sample data

The sample pack demonstrates the expected folder layout (`vector/`, `raster/imagery-default-mode/`, `raster/imagery-legacy-mode/`, and `flags-demo/`) together with ready-to-run configuration files (`demo_config_default.py` and `demo_config_legacy.py`).

### 1. Download and unpack the demo data

```bash
# ~160 MB download
curl -L -o tsbrowser_demo_setup.zip \
  https://nxc.joanneum.at/index.php/s/TKGAdd4FJ99LDgy/download
unzip tsbrowser_demo_setup.zip
```

This creates a `tsbrowser_demo_setup/` directory containing the raster stacks, sample vector file, and example configs.

### 2. Run the default-mode example

```bash
rm -rf tsbrowser_demo_setup/flags-demo/  # start fresh
cd tsbrowser_demo_setup
uv run tsbrowser demo_config_default.py
```

This configuration expects on-the-fly QA evaluation and points the CLI to the `vector/demo_points.shp` shapefile together with Sentinel-2 L2A chips stored in `raster/imagery-default-mode/`. Use `--pid <id>` to open specific samples or leave it out to iterate through all unflagged features.

<p align="center">
<img src="docs/img/demo.png" width="1000">
</p>

### 3. Try the legacy-mode example

```bash
uv run tsbrowser /path/to/tsbrowser_demo_setup/demo_config_legacy.py
```

Legacy mode works with the pre-filtered chips under `raster/imagery-legacy-mode/` and skips QA masking.

### 4. Inspect the outputs

Flag annotations are written to the directory defined by `flag_dir` in each config (`tsbrowser_demo_setup/flags-demo/` by default). Copy one of the demo configs if you want to modify paths, interpreter IDs, or quality settings:

```bash
cp tsbrowser_demo_setup/demo_config_default.py tsbrowser_demo_setup/my_config.py
uv run tsbrowser tsbrowser_demo_setup/my_config.py
```

Make sure the relative paths inside the config still point to the folders that contain your data.
