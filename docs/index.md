# Introduction

`tsbrowser` is a Python utility to visualize and label satellite data time-series.

It provides an interactive plot in which segments and events can be labelled. The plot provides 2D time series, image chip visualization as well as integration with ESRI wayback machine for very high resolution imagery.

## Installation

The package can be installed directly from github. Since the package provides a script entrypoint, it can also be run directly using `uvx` or `pipx`.

=== "pip"
    ```
    pip install git+https://github.com/JR-DIGITAL/tsbrowser.git
    tsbrowser
    ```

=== "pipx"
    ```
    pipx run --spec git+https://github.com/JR-DIGITAL/tsbrowser.git tsbrowser
    ```

=== "uvx"
    ```
    uvx git+https://github.com/JR-DIGITAL/tsbrowser.git
    ```

This should run the `tsbrowser` command line interface and print the following output:

```
usage: tsbrowser [-h] [--pid [STR or INT ...]] [--pattern STR] [--semilogy] [--scalewindow FLOAT FLOAT] [--startdate YYYYMMDD] [--stopdate YYYYMMDD] [--preload-threads INT] PATH
tsbrowser: error: the following arguments are required: PATH
```
