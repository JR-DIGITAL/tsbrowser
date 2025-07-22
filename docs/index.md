# Introduction

`tsbrowser` is a Python utility to visualize and label satellite data time-series.

It provides an interactive plot in which segments and events can be labelled. The plot provides 2D time series, image chip visualization as well as integration with ESRI wayback machine for very high resolution imagery.

## Installation

The package can be installed directly from github. Since the package provides a script entrypoint, it can also be run directly using `uvx` or `pipx`.

=== "pip"
    ```
    pip install git+https://github.com/jonasViehweger/s2-timeseries-labeller.git
    tsbrowser
    ```

=== "pipx"
    ```
    pipx run --spec git+https://github.com/jonasViehweger/s2-timeseries-labeller.git tsbrowser
    ```

=== "uvx"
    ```
    uvx git+https://github.com/jonasViehweger/s2-timeseries-labeller.git
    ```

This should run the `tsbrowser` command line interface and print the following output:

```
usage: tsbrowser [-h] [--pid [STR or INT ...]] [--pattern STR] [--semilogy] [--scalewindow FLOAT FLOAT] [--startdate YYYYMMDD] [--stopdate YYYYMMDD] [--preload-threads INT] PATH
tsbrowser: error: the following arguments are required: PATH
```

## Setup

To run the package, the following prerequisites need to be met

1. Input data available
2. Shapefile with sample points
3. Prepared tsbrowser configuration file

### Input data

To run the package, stacks of satellite data chips are required. Have a look at the example data provided with the package at TODO on how to structure this data.

### Sample Shapefile

The sample shapefile provides the actual sample points which should be interpreted. To give the `tsbrowser` utility all the necessary information, a few fields in the attribute table need to be specified per sample:

- ID: unique ID of the sample
- quality file location for the sample (folder)
- raster file location for the sample (folder)

The exact names of the columns need to be given in the tsbrowser configuration file

### `tsbrowser` configuration

The `tsbrowser` configuration specifies how the tsbrowser instance should behave. In it, the input files are specified as well as what kind of data will be shown in the interpretation interface.

??? Configuration

    ```python title="tsbrowser configuration file"
    --8<-- "tsbrowser_test_config.py"
    ```

## Running `tsbrowser`

Once all the prerequisites are met, tsbrowser can be used to carry out interpretation of time-series.

The utility has different modes of operation. These depend on how the utility is called. 

```
tsbrowser /path/to/tsbrowser_config.py --pid 1 2 3
```

The `--pid` argument in the CLI allows to specify a single sample ID or a list of sample IDs from the sample shapefile which is specified in the given `tsbrowser` configuration file. If multiple IDs are given, they are shown sequentially to the user. In the example above, the sample IDs 1, 2 and 3 will be shown sequentially to the interpreter.


However the default mode is an automatic orchestrator. It is run when no `--pid` is given. 

```
tsbrowser /path/to/tsbrowser_config.py
```

This mode compares which samples are in the sample shapefile and which ones were already interpreted (have a flag file in the `flag_dir` specified in the config file). This mode then sequentially shows samples from the sample vector file which have not been interpreted yet.



