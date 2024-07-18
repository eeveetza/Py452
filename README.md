# Python Implementation of Recommendation ITU-R P.452

<!--This is development code!-->

This code repository contains a python software implementation of [Recommendation ITU-R P.452-18](https://www.itu.int/rec/R-REC-P.452/en) with a prediction procedure for the evaluation of interference between stations on the surface of the Earth at frequencies above about 100 MHz.  

<!--This development version implements the clutter loss model along the path profile.

This is a development code and it is not necessarily in line with the original reference [MATLAB/Octave Implementation of Recommendation ITU-R P.452](https://github/eeveetza/p452) approved by ITU-R Working Party 3M and published on [ITU-R SG 3 Software, Data, and Validation Web Page](https://www.itu.int/en/ITU-R/study-groups/rsg3/Pages/iono-tropo-spheric.aspx).
-->
This code is functionally identical to the original reference [MATLAB/Octave Implementation of Recommendation ITU-R P.452](https://github/eeveetza/p452) approved by ITU-R Working Party 3M and published on [ITU-R SG 3 Software, Data, and Validation Web Page](https://www.itu.int/en/ITU-R/study-groups/rsg3/Pages/iono-tropo-spheric.aspx).

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
|`/src/Py452/P452.py`                | python implementation of Recommendation ITU-R P.452-18         |
|`/tests/validateP452.py`          | python script used to validate the implementation of Recommendation ITU-R P.452-18 in `P452.bt_loss()`             |
|`/tests/validation_examples/profiles/`    | Folder containing a proposed set of terrain profiles for validation of python implementation (or any other software implementation) of this Recommendation |
|`/tests/validation_examples/results/`	   | Folder containing a proposed set of input parameters and the intermediate and final results for the set of terrain profiles defined in the folder `./validation_profiles/` |


## Function Call

~~~ 
    Lb = P452.bt_loss(f, p, d, h, g, zone, htg, hrg, phit_e, phit_n, phir_e, phir_n, Gt, Gr, pol, dct, dcr, press, temp)
~~~


## Required input arguments of function `P452.bt_loss`

| Variable          | Type   | Units | Limits       | Description  |
|-------------------|--------|-------|--------------|--------------|
| `f`               | scalar double | GHz   | ~0.1 ≤ `f` ≤ ~50 | Frequency   |
| `p         `      | scalar double | %     | 0.001 ≤ `p` ≤ 50 | Time percentage for which the calculated basic transmission loss is not exceeded |
| `d`               | array double | km    |  0 < `max(d)` ≤ ~10000 | Terrain profile distances (in the ascending order from the transmitter)|
| `h`          | array double | m (asl)   |   | Terrain profile heights |
| `g`          | array double | m (asl)   |  | Clutter + Terrain profile heights   |
| `zone`           | array int    |       | 1 - Coastal Land, 2 - Inland, 3 - Sea             |  Radio-climatic zone types |
| `htg`           | scalar double    | m      |           |  Tx antenna height above ground level |
| `hrg`           | scalar double    | m      |          |  Rx antenna height above ground level |
| `phit_e`           | scalar double    | deg      |     0 ≤ `phit_e`  ≤ 360          |  Tx longitude |
| `phit_n`           | scalar double    | deg      |     -90 ≤ `phit_n`  ≤ 90          |  Tx latitude |
| `phir_e`           | scalar double    | deg      |     0 ≤ `phir_e`  ≤ 360          |  Rx longitude |
| `phir_n`           | scalar double    | deg      |     -90 ≤ `phir_n`  ≤ 90          |  Rx latitude |
| `Gt`  `Gr`           | scalar double  |   dBi    |           |  Tx/Rx antenna gain in the direction of the horizon towards along the great-circle interference path. |
| `pol`           | scalar int    |       |   `pol`  = 1, 2          |  Polarization of the signal: 1 - horizontal, 2 - vertical |
| `dct`           | scalar double    | km      |   `dct` ≥ 0          |  Distance over land from the Tx antenna to the coast along the great-circle interference path. To be set to zero for a terminal on a ship or sea platform.|
| `dcr`           | scalar double    | km      |   `dcr` ≥ 0          |  Distance over land from the Rx antenna to the coast along the great-circle interference path. To be set to zero for a terminal on a ship or sea platform.|
| `press`           | scalar double    | hPa      |             | Dry air pressure.|
| `temp`           | scalar double    | deg C      |             | Air temperature.|



 
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
