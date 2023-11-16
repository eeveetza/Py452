# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,line-too-long,too-many-lines,too-many-arguments,too-many-locals,too-many-statements
"""
  This script is used to validate the python implementation of 
  Recommendation ITU-R P.452 as defined in the package Py452
  
  Author: Ivica Stevanovic (IS), Federal Office of Communications, Switzerland
  Revision History:
  Date            Revision
  30Mar2022       Initial version (IS)  
  09NOV2023       Revised using Pandas (IS, following validatePy2001.py (Adrien Demarez))
  16NOV2023       Aligned with ITU-R P.452-18
"""

import os
import numpy as np
import pandas as pd

from Py452 import P452

tol = 1e-6
success = 0
total = 0

# path to the folder containing test profiles
test_profiles = "./validation_examples/profiles/"
test_results = "./validation_examples/results/"

# begin code
# Collect all the filenames .csv in the folder test_profiles that contain the profile data
try:
    filenames = [f for f in os.listdir(test_profiles) if f.endswith(".csv")]
except:
    print("The system cannot find the given folder " + test_profiles)

print("\n")

for filename in filenames:
    
    # read the path profiles, input arguments and reference values
    
    df1 = pd.read_csv(test_profiles + filename, skiprows=1, names=['d', 'h', 'r', 'zonestr', 'zone'])
    df2 = pd.read_csv(test_results + filename.replace("test_profile", "test_result"), skiprows = 1, 
                    names = ['profile', 'f', 'p', 'htg', 'hrg', 'phit_e', 'phit_n', 'phir_e', 'phir_n', 'Gt', 'Gr', 'pol', 'dct', 'dcr', 
                            'press', 'temp', 'ae', 'dtot', 	
                            'hts', 'hrs', 'theta_t', 'theta_r', 'theta', 'hm', 'hte', 'hre', 'hstd', 'hsrd',
                            'dlt', 'dlr', 'path', 'dtm', 'dlm', 'b0', 'omega', 'DN', 'N0', 'Lb', 'Lbfsg', 'Lb0p', 'Lb0b', 'Ldsph', 'Ld50', 'Ldp', 'Lbs', 'Lba'])
    
    d = df1.d.to_numpy()
    h = df1.h.to_numpy()
    r = df1.r.to_numpy()
    zone =  df1.zone.to_numpy()
    
    g = h + r
    
        # Ensure that vector d is ascending
    if not np.all(np.diff(d) >= 0):
        raise ValueError("The array of path profile points d[i] must be in ascending order.")

    # Ensure that d[0] = 0 (Tx position)
    if d[0] > 0.0:
        raise ValueError("The first path profile point d[0] = " + str(d[0]) + " must be zero.")

    dtot = d[-1]

    # Apply the condition in Step 4: Radio profile 
    # gi is the terrain height in metres above sea level for all the points at a distance from transmitter or receiver less than 50 m.

    (kk, ) = np.where(d < 50/1000)
    if (~P452.isempty(kk)):
        g[kk] = h[kk]
    
    (kk,  ) = np.where(dtot - d < 50/1000)
    if (~P452.isempty(kk)):
        g[kk] = h[kk]
    
    print("Processing file " + filename + "\n")
    
    failed = False
    
    (nrows, ) = d.shape
    
    # sort the reference values in an array
    row = df2.iloc[0]
    
    # Compute  dtm     -   the longest continuous land (inland + coastal) section of the great-circle path (km)
    zone_r = 12
    dtm = P452.longest_cont_dist(d, zone, zone_r)

    # Compute  dlm     -   the longest continuous inland section of the great-circle path (km)
    zone_r = 2
    dlm = P452.longest_cont_dist(d, zone, zone_r)

# Calculate the longitude and latitude of the mid-point of the path, phim_e,
    # and phim_n for dpnt = 0.5dt
    Re = 6371
    dpnt = 0.5 * dtot
    phim_e, phim_n, _, _ = P452.great_circle_path(row.phir_e, row.phit_e, row.phir_n, row.phit_n, Re, dpnt)


    # Find radio-refractivity lapse rate dN 
    # using the digital maps at phim_e (lon), phim_n (lat) - as a bilinear interpolation
    DN50 = P452.DigitalMaps["DN50"]
    N050 = P452.DigitalMaps["N050"]
    
    DN = P452.interp2(DN50, phim_e, phim_n, 1.5, 1.5)
    N0 = P452.interp2(N050, phim_e, phim_n, 1.5, 1.5)
    
    # Compute b0
    b0 = P452.beta0(phim_n, dtm, dlm)

    ae, ab = P452.earth_rad_eff(DN)

    hst, hsr, hstd, hsrd, hte, hre, hm, dlt, dlr, theta_t, theta_r, theta, pathtype = P452.smooth_earth_heights(d, h, row.htg, row.hrg, ae, row.f)

    # Tx and Rx antenna heights above mean sea level amsl (m)
    hts = h[0] + row.htg
    hrs = h[-1] + row.hrg

    # Compute the path fraction over see

    omega = P452.path_fraction(d, zone, 3)

    # verify the results struct `out` against the reference struct `ppref`
    out = np.zeros(19)
    pref = np.zeros(19)
    outstr = np.empty(19, dtype=object)
    

    pref[0] = row.ae
    pref[1] = row.dtot
    pref[2] = row.hts
    pref[3] = row.hrs
    pref[4] = row.theta_t
    pref[5] = row.theta_r
    pref[6] = row.theta
    pref[7] = row.hm
    pref[8] = row.hte
    pref[9] = row.hre
    pref[10] = row.hstd
    pref[11] = row.hsrd
    pref[12] = row.dlt
    pref[13] = row.dlr
    pref[14] = (1 if row.path.lower() == 'line of sight' else 2) 
    pref[15] = row.dtm
    pref[16] = row.dlm
    pref[17] = row.b0
    pref[18] = row.omega
    
    out[0] = ae
    out[1] = dtot
    out[2] = hts
    out[3] = hrs
    out[4] = theta_t
    out[5] = theta_r
    out[6] = theta
    out[7] = hm
    out[8] = hte
    out[9] = hre
    out[10] = hstd
    out[11] = hsrd
    out[12] = dlt
    out[13] = dlr
    out[14] = pathtype
    out[15] = dtm
    out[16] = dlm
    out[17] = b0
    out[18] = omega

    outstr[0] = "ae"
    outstr[1] = "dtot"
    outstr[2] = "hts"
    outstr[3] = "hrs"
    outstr[4] = "theta_t"
    outstr[5] = "theta_r"
    outstr[6] = "theta"
    outstr[7] = "hm"
    outstr[8] = "hte"
    outstr[9] = "hre"
    outstr[10] = "hstd"
    outstr[11] = "hsrd"
    outstr[12] = "dlt"
    outstr[13] = "dlr"
    outstr[14] = "pathtype"
    outstr[15] = "dtm"
    outstr[16] = "dlm"
    outstr[17] = "b0"
    outstr[18] = "omega"

    # verify the results `out` against the reference `pref`

    for i in range(0, len(out)):
        # print('%s: ref: %g,   comp: %g\n' %(outstr[i], pref[i], out[i]) )
        error = np.abs(out[i] - pref[i])

        if error > tol:
            print("Error in parameter %s larger than tolerance %g: %g\n" % (outstr[i], tol, error))
            failed = True

    # extract reference transmission losses
    Lb_ref = df2.Lb.to_numpy()
    Lbfsg_ref = df2.Lbfsg.to_numpy()
    Lb0p_ref = df2.Lb0p.to_numpy()
    Lb0b_ref = df2.Lb0b.to_numpy()
    Ldsph_ref = df2.Ldsph.to_numpy()
    Ld50_ref = df2.Ld50.to_numpy()
    Ldp_ref = df2.Ldp.to_numpy()
    Lbs_ref = df2.Lbs.to_numpy()
    Lba_ref = df2.Lba.to_numpy()

    # compute the transmission losses using python package
    Lbfsg = np.zeros(nrows)
    Lb0p = np.zeros(nrows)
    Lb0b = np.zeros(nrows)
    Lbs = np.zeros(nrows)
    Lba = np.zeros(nrows)
    Lbulla = np.zeros(nrows)
    Lbulls = np.zeros(nrows)
    Ldsph = np.zeros(nrows)
    Ld = np.zeros(nrows)
    Ldp = np.zeros(nrows)
    Ld50 = np.zeros(nrows)
    Lb = np.zeros(nrows)

    out1 = np.zeros((9, nrows))
    out1str = np.empty(9, dtype=object)

    for i, row in df2.iterrows():
        d3D = np.sqrt(dtot**2.0 + (hts - hrs) ** 2 / 1e6)
        Lbfsg[i], Lb0p[i], Lb0b[i] = P452.pl_los(d3D, row.f, row.p, b0, omega, row.temp, row.press, dlt, dlr)

        Lbs[i] = P452.tl_tropo(dtot, theta, row.f, row.p, row.temp, row.press, N0, row.Gt, row.Gr)

        Lba[i] = P452.tl_anomalous(dtot, dlt, dlr, row.dct, row.dcr, dlm, hts, hrs, hte, hre, hm, theta_t, theta_r, row.f, row.p, row.temp, row.press, omega, ae, b0)

        Lbulla[i] = P452.dl_bull(d, g, hts, hrs, ae, row.f)

        # Use the method in 4.2.1 for a second time, with all profile heights hi
        # set to zero and modified antenna heights given by

        hts1 = hts - hstd  # eq (38a)
        hrs1 = hrs - hsrd  # eq (38b)
        h1 = np.zeros(h.shape)

        # where hstd and hsrd are given in 5.1.6.3 of Attachment 2. Set the
        # resulting Bullington diffraction loss for this smooth path to Lbulls

        Lbulls[i] = P452.dl_bull(d, h1, hts1, hrs1, ae, row.f)

        # Use the method in 4.2.2 to radiomaps the spherical-Earth diffraction loss
        # for the actual path length (dtot) with

        hte1 = hts1  # eq (39a)
        hre1 = hrs1  # eq (39b)

        Ldsph_pol = P452.dl_se(dtot, hte1, hre1, ae, row.f, omega)
        Ldsph[i] = Ldsph_pol[int(row.pol - 1)]

        # Diffraction loss for the general path is now given by
        
        Ld[i] = Lbulla[i] + max(Ldsph[i] - Lbulls[i], 0)  # eq (40)

        Ldp_pol, Ld50_pol = P452.dl_p(d, g, hts, hrs, hstd, hsrd, row.f, omega, row.p, b0, DN)

        Ldp[i] = Ldp_pol[int(row.pol - 1)]
        Ld50[i] = Ld50_pol[int(row.pol - 1)]

        Lb[i] = P452.bt_loss(row.f, row.p, d, h, g, zone, row.htg, row.hrg, row.phit_e, row.phit_n, row.phir_e, row.phir_n, row.Gt, row.Gr, row.pol, row.dct, row.dcr,  row.press, row.temp)

        out1[0, i] = Lbfsg[i] - Lbfsg_ref[i]
        out1str[0] = "Lbfsg"

        out1[1, i] = Lb0p[i] - Lb0p_ref[i]
        out1str[1] = "Lb0p"

        out1[2, i] = Lb0b[i] - Lb0b_ref[i]
        out1str[2] = "Lb0b"

        out1[3, i] = Ldsph[i] - Ldsph_ref[i]
        out1str[3] = "Ldsph"

        out1[4, i] = Ld50[i] - Ld50_ref[i]
        out1str[4] = "Ld50"

        out1[5, i] = Ldp[i] - Ldp_ref[i]
        out1str[5] = "Ldp"

        out1[6, i] = Lbs[i] - Lbs_ref[i]
        out1str[6] = "Lbs"

        out1[7, i] = Lba[i] - Lba_ref[i]
        out1str[7] = "Lba"

        out1[8, i] = Lb[i] - Lb_ref[i]
        out1str[8] = "Lb"

    # verify error in the results out1 against tolarance

    for i in range(0, len(out1str)):
        maxi = max(np.abs(out1[i, :])) 

        (kk,) = (np.atleast_1d(maxi > tol).nonzero())
        
        if ~P452.isempty(kk):
            for kki in range(0, len(kk)):
                print("\tMaximum deviation found in %s larger than tolerance %g:  %g\n" % (out1str[i], tol, maxi))
                failed = True

    if not failed:
        success += 1

    total +=  1
print("----------------------------------------------------------\n")
print("Validation results: %d out of %d tests passed successfully.\n" % (success, total))
if success == total:
    print("The deviation from the reference results is smaller than %g.\n" % (tol))
