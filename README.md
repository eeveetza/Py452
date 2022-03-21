# Python Implementation of Recommendation ITU-R P.452

This code repository contains a python software implementation of  [Recommendation ITU-R P.452-17](https://www.itu.int/rec/R-REC-P.452/en)  with a prediction procedure for the evaluation of interference between stations on the surface of the Earth at frequencies above about 0.1 GHz.  

This is a translation of the original reference [MATLAB/Octave Implementation of Recommendation ITU-R P.452](https://github/eeveetza/p452) approved by ITU-R Working Party 3M and published by Study Group 3 on [ITU-R SG 3 Software, Data, and Validation Web Page](https://www.itu.int/en/ITU-R/study-groups/rsg3/Pages/iono-tropo-spheric.aspx).

The package can be downloaded and installed using:
~~~
python -m pip install "git+https://github.com/eeveetza/Py452/#egg=Py452"   
~~~

and imported as follows
~~~
from Py452 import P452
~~~

| File/Folder               | Description                                                         |
|----------------------------|---------------------------------------------------------------------|
|`/src/Py452/P452.py`                | python implementation of Recommendation ITU-R P.452-17         |
|`/tests/validateP452.py`          | python script used to validate the implementation of Recommendation ITU-R P.452-17 in `P452.bt_loss()`             |
|`/tests/validation_examples/profiles/`    | Folder containing a proposed set of terrain profiles for validation of python implementation (or any other software implementation) of this Recommendation |
|`/tests/validation_examples/results/`	   | Folder containing a proposed set of input parameters and the intermediate and final results for the set of terrain profiles defined in the folder `./validation_profiles/` |


## Function Call

The function `P452.bt_loss` can be called

1. by invoking only the required input arguments:
~~~ 
    Lb = P452.bt_loss(f, p, d, h, zone, htg, hrg, phi_path, Gt, Gr, pol, dct, dcr, DN, N0, press, temp)
~~~
1. by explicitly invoking all the input arguments (including the optional arguments related to clutter):
~~~
    Lb = P452.bt_loss(f, p, d, h, zone, htg, hrg, phi_path, Gt, Gr, pol, dct, dcr, DN, N0, press, temp, \
                      ha_t, ha_r, dk_t, dk_r)
~~~ 

## Required input arguments of function `P452.bt_loss`

| Variable          | Type   | Units | Limits       | Description  |
|-------------------|--------|-------|--------------|--------------|
| `f`               | scalar double | GHz   | ~0.1 ≤ `f` ≤ ~50 | Frequency   |
| `p         `      | scalar double | %     | 0.001 ≤ `p` ≤ 50 | Time percentage for which the calculated basic transmission loss is not exceeded |
| `d`               | array double | km    |  0 < `max(d)` ≤ ~10000 | Terrain profile distances (in the ascending order from the transmitter)|
| `h`          | array double | m (asl)   |   | Terrain profile heights |
| `zone`           | array int    |       | 1 - Coastal Land, 2 - Inland, 3 - Sea             |  Radio-climatic zone types |
| `htg`           | scalar double    | m      |           |  Tx antenna height above ground level |
| `hrg`           | scalar double    | m      |          |  Rx antenna height above ground level |
| `phi_path`           | scalar double    | deg      |   -90 ≤ `phi_t`  ≤ 90          |  Latitude of path center between Tx and Rx stations |
| `Gt`  `Gr`           | scalar double  |   dBi    |           |  Tx/Rx antenna gain in the direction of the horizon towards along the great-circle interference path. |
| `pol`           | scalar int    |       |   `pol`  = 1, 2          |  Polarization of the signal: 1 - horizontal, 2 - vertical |
| `dct`           | scalar double    | km      |   `dct` ≥ 0          |  Distance over land from the Tx antenna to the coast along the great-circle interference path. To be set to zero for a terminal on a ship or sea platform.|
| `dcr`           | scalar double    | km      |   `dcr` ≥ 0          |  Distance over land from the Rx antenna to the coast along the great-circle interference path. To be set to zero for a terminal on a ship or sea platform.|
| `DN`            | scalar double    | N-units/km      | `DN`> 0           | The average radio-refractive index lapse-rate through the lowest 1 km of the atmosphere at the path-center. It can be derived from an appropriate map.  |
| `N0`           | scalar double    | N-units      |             | The sea-level surface refractivity at the path-centre. It can be derived from an appropriate map.|
| `press`           | scalar double    | hPa      |             | Dry air pressure.|
| `temp`           | scalar double    | deg C      |             | Air temperature.|

## Optional input arguments of function `P452.bt_loss`
| Variable          | Type   | Units | Limits       | Description  |
|-------------------|--------|-------|--------------|--------------|
| `ha_t`           | scalar double    | m      |             | Clutter nominal height at the Tx side |
| `ha_r`           | scalar double    | m      |             | Clutter nominal height at the Rx side |
| `dk_t`           | scalar double    | km      |             | Clutter nominal distance at the Tx side |
| `dk_r`           | scalar double    | km      |             | Clutter nominal distance at the Rx side |



 
## Outputs ##

| Variable   | Type   | Units | Description |
|------------|--------|-------|-------------|
| `Lb`    | double | dB    | Basic transmission loss |



## Software Versions
The code was tested and runs on:
* python3.9

## References

* [Recommendation ITU-R P.452](https://www.itu.int/rec/R-REC-P.452/en)

* [ITU-R SG 3 Software, Data, and Validation Web Page](https://www.itu.int/en/ITU-R/study-groups/rsg3/Pages/iono-tropo-spheric.aspx)

* [MATLAB/Octave Implementation of Recommendation ITU-R P.452](https://github/eeveetza/p452)
