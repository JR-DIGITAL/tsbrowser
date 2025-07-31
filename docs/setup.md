# Setup

To run the package, the following prerequisites need to be met:

1. Input imagery available
2. Vector file with sample points
3. Prepared tsbrowser configuration file

## Input imagery

To run the package, time series of satellite imagery are required. Have a look at the demonstration data provided [here](https://nxc.joanneum.at/index.php/s/TKGAdd4FJ99LDgy) to learn how to structure these data. The tool supports two ways of data preparation:

1. Default mode
2. Legacy mode

In default mode, you need to prepare the imagery together with corresponding quality assessment (QA) layers, typically cloud masks. The tool then evaluates the QA layers on-the-fly in order to plot clean time series. The second option, legacy mode, bypasses on-the-fly evaluation and therefore requires pre-filtered imagery. The demo data set contains examples for both operating modes in the *raster* directory.

## Vector file with sample points

The sample vector file provides the actual sample points which should be interpreted. To give the `tsbrowser` utility all the necessary information, a few fields in the attribute table need to be specified per sample:

- ID: unique ID of the sample (`int` or `str`)
- QA raster file directory for the sample (`str`, path either absolute or relative to the vector file directory)
- Imagery raster file directory for the sample (`str`, path either absolute or relative to the vector file directory)

The exact names of the respective attributes need to be given in the `tsbrowser` configuration file. The [demonstration data set](https://nxc.joanneum.at/index.php/s/TKGAdd4FJ99LDgy) includes a suitable shapefile in the *vector* directory.

## `tsbrowser` configuration file

The `tsbrowser` configuration file includes all the available parameters of the tool. Please refer to the comments given for each parameter for further details. The configuration of `tsbrowser` depends strongly on locally available input data and its specific properties. Therefore, there is no default configuration that will work out-of-the-box. The configuration file needs to be set up for each use case. If interpretation is carried out by multiple users, we recommend to split the sample points into several partitions and create configurations per user. 

??? Configuration

    ```python title="tsbrowser configuration file"
    --8<-- "tsbrowser_example_config.py"
    ```
