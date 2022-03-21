# -*- coding: utf-8 -*-
"""
  This script is used to validate the python implementation of 
  Recommendation ITU-R P.452 as defined in the package Py452
  
  Author: Ivica Stevanovic (IS), Federal Office of Communications, Switzerland
  Revision History:
  Date            Revision
  30Mar2022       Initial version (IS)  
  
"""
import csv
import os
from traceback import print_last
import numpy as np

from Py452 import P452

tol = 1e-6
success = 0
total = 0

# path to the folder containing test profiles
test_profiles = './validation_examples/profiles/'
test_results  = './validation_examples/results/'


# begin code
# Collect all the filenames .csv in the folder test_profiles that contain the profile data
try:
    filenames = [f for f in os.listdir(test_profiles) if f.endswith('.csv')]
except:
    print ("The system cannot find the given folder " + test_profiles)
    
for filename1 in filenames:
    
    print ('***********************************************\n')
    print ('Processing file '  + filename1 + '\n')
    print ('***********************************************\n')
    
    failed = False
    
    # read the path profiles
    
    rows = []
    with open(test_profiles + filename1, 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            rows.append(row)
            
    x = np.mat(rows)
    
    d = np.double(x[:,0])
    d = np.squeeze(np.asarray(d))
    h = np.double(x[:,1])
    h = np.squeeze(np.asarray(h))
    zone = np.double(x[:,3])
    zone = np.squeeze(np.asarray(zone))
    

    # read the input arguments and reference values
    fname_part = filename1[12:]
    test_result = 'test_result' + fname_part
    
    rows = []
    with open(test_results + test_result, 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            rows.append(row)
            
    x = np.mat(rows)
    (nrows, ncols) = x.shape
    ff = np.zeros(nrows)
    pp = np.zeros(nrows)
    
    for i in range(0, nrows):
        ff[i] = np.double(x[i,1])
        pp[i] = np.double(x[i,2])
    
    htg    = np.double( x[0, 3] )
    hrg    = np.double( x[0, 4] )
    phi_path  = np.double( x[0, 5] )
    Gt     = np.double( x[0, 6] )
    Gr     = np.double( x[0, 7] )
    pol    = np.double( x[0, 8] )
    dct    = np.double( x[0, 9] )
    dcr    = np.double( x[0,10] )
    DN     = np.double( x[0,11] ) 
    N0     = np.double( x[0,12] )
    press  = np.double( x[0,13] )
    temp   = np.double( x[0,14] )
    ha_t   = np.double( x[0,15] )
    ha_r   = np.double( x[0,16] )
    dk_t   = np.double( x[0,17] )
    dk_r   = np.double( x[0,18] )
    
    # collect reference values
    
    pref = np.zeros(19)
    
    for i in range(0,19):
        if i == 14:
            path = x[0,33]    
            if path.lower() == 'line of sight' :
                pathtype = 1
            else:
                pathtype = 2
            pref[i] = pathtype
        else:
            
            pref[i] = np.double( x[0,i+19] )
            
        
    # sort the reference values in an array
        
    dc, hc, zonec, htgc, hrgc, Aht, Ahr = P452.closs_corr(ff[0], d, h, zone, htg, hrg, ha_t, ha_r, dk_t, dk_r)

   
    # Compute  dtm     -   the longest continuous land (inland + coastal) section of the great-circle path (km)
    zone_r = 12
    dtm = P452.longest_cont_dist(d, zone, zone_r)
    
    # Compute  dlm     -   the longest continuous inland section of the great-circle path (km)
    zone_r = 2
    dlm = P452.longest_cont_dist(d, zone, zone_r)
    
    # Compute b0
    b0 = P452.beta0(phi_path, dtm, dlm)
    
    ae, ab = P452.earth_rad_eff(DN)
    
    hst, hsr, hstd, hsrd, hte, hre, hm, dlt, dlr, theta_t, theta_r, theta, pathtype = P452.smooth_earth_heights(dc, hc, htgc, hrgc, ae, ff[0])
    
    dtot = dc[-1]-dc[0]
    
    #Tx and Rx antenna heights above mean sea level amsl (m)
    hts = hc[0]  + htgc
    hrs = hc[-1] + hrgc
    
    # Compute the path fraction over see
    
    omega = P452.path_fraction(d, zone, 3)
    
    
    # verify the results struct `out` against the reference struct `ppref`
    out = np.zeros(19)
    outstr = np.empty(19, dtype = object)
    
    out[0]  = ae
    out[1]  = dtot   
    out[2]  = hts    
    out[3]  = hrs    
    out[4]  = theta_t
    out[5]  = theta_r
    out[6]  = theta  
    out[7]  = hm     
    out[8]  = hte    
    out[9]  = hre    
    out[10] = hstd   
    out[11] = hsrd   
    out[12] = dlt    
    out[13] = dlr    
    out[14] = pathtype   
    out[15] = dtm    
    out[16] = dlm    
    out[17] = b0     
    out[18] = omega  

    outstr[0]  = 'ae'
    outstr[1]  = 'dtot'   
    outstr[2]  = 'hts'   
    outstr[3]  = 'hrs'    
    outstr[4]  = 'theta_t'
    outstr[5]  = 'theta_r'
    outstr[6]  = 'theta'  
    outstr[7]  = 'hm'     
    outstr[8]  = 'hte'    
    outstr[9]  = 'hre'    
    outstr[10] = 'hstd'   
    outstr[11] = 'hsrd'   
    outstr[12] = 'dlt'   
    outstr[13] = 'dlr'    
    outstr[14] = 'pathtype'
    outstr[15] = 'dtm'    
    outstr[16] = 'dlm'    
    outstr[17] = 'b0'     
    outstr[18] = 'omega'  


    # verify the results `out` against the reference `pref`
    
    
    for i in range(0, len(out)) :
        #print('%s: ref: %g,   comp: %g\n' %(outstr[i], pref[i], out[i]) )
        error = np.abs( out[i] - pref[i] )
        
        if error > tol:
            print('Error in parameter %s larger than tolerance %g: %g\n' % (outstr[i], tol, error) )
            failed = True

            
        

           
    # extract reference transmission losses
    Lb_ref    = np.zeros(nrows)
    Lbfsg_ref = np.zeros(nrows)
    Lb0p_ref  = np.zeros(nrows)
    Lb0b_ref  = np.zeros(nrows)
    Ldsph_ref = np.zeros(nrows)
    Ld50_ref  = np.zeros(nrows)
    Ldp_ref   = np.zeros(nrows)
    Lbs_ref   = np.zeros(nrows)
    Lba_ref   = np.zeros(nrows)
    
    for i in range(0, nrows):
        Lb_ref[i]    = np.double( x[i,38] )
        Lbfsg_ref[i] = np.double( x[i,39] )
        Lb0p_ref[i]  = np.double( x[i,40] )
        Lb0b_ref[i]  = np.double( x[i,41] )
        Ldsph_ref[i] = np.double( x[i,42] )
        Ld50_ref[i]  = np.double( x[i,43] )
        Ldp_ref[i]   = np.double( x[i,44] )
        Lbs_ref[i]   = np.double( x[i,45] )
        Lba_ref[i]   = np.double( x[i,46] )

    # compute the transmission losses using MATLAB functions
    Lbfsg   = np.zeros(nrows)
    Lb0p    = np.zeros(nrows)
    Lb0b    = np.zeros(nrows)
    Lbs     = np.zeros(nrows)
    Lba     = np.zeros(nrows)
    Lbulla  = np.zeros(nrows)
    Lbulls  = np.zeros(nrows)
    Ldsph   = np.zeros(nrows)
    Ld      = np.zeros(nrows)
    Ldp     = np.zeros(nrows)
    Ld50    = np.zeros(nrows)
    Lb      = np.zeros(nrows)
    
    
    out1 = np.zeros((9,nrows))
    out1str = np.empty(9,  dtype = object)
    
    for i in range(0,nrows):
        d3D = np.sqrt(dtot**2.0 + (hts-hrs)**2/1e6)
        Lbfsg[i], Lb0p[i], Lb0b[i] = P452.pl_los(d3D, \
            ff[i], \
            pp[i], \
            b0, \
            omega, \
            temp, \
            press, \
            dlt, \
            dlr)
        
        Lbs[i] = P452.tl_tropo(dtot, \
            theta, \
            ff[i], \
            pp[i], \
            temp, \
            press, \
            N0, \
            Gt, \
            Gr )
        
        Lba[i] = P452.tl_anomalous(dtot, \
            dlt, \
            dlr, \
            dct, \
            dcr, \
            dlm, \
            hts, \
            hrs, \
            hte, \
            hre, \
            hm, \
            theta_t, \
            theta_r, \
            ff[i], \
            pp[i], \
            temp, \
            press, \
            omega, \
            ae, \
            b0)
        
        Lbulla[i] = P452.dl_bull(dc, hc, hts, hrs, ae, ff[i])
        
        # Use the method in 4.2.1 for a second time, with all profile heights hi
        # set to zero and modified antenna heights given by
        
        hts1 = hts - hstd   # eq (38a)
        hrs1 = hrs - hsrd   # eq (38b)
        h1 = np.zeros(hc.shape)
        
        # where hstd and hsrd are given in 5.1.6.3 of Attachment 2. Set the
        # resulting Bullington diffraction loss for this smooth path to Lbulls
        
        Lbulls[i] = P452.dl_bull(dc, h1, hts1, hrs1, ae, ff[i])
        
        # Use the method in 4.2.2 to radiomaps the spherical-Earth diffraction loss
        # for the actual path length (dtot) with
        
        hte1 = hts1             # eq (39a)
        hre1 = hrs1             # eq (39b)
        
        Ldsph_pol = P452.dl_se(dtot, hte1, hre1, ae, ff[i], omega)
        Ldsph[i] = Ldsph_pol[int(pol-1)]
          
        # Diffraction loss for the general path is now given by
        Ld[i] = Lbulla[i] + max( Ldsph[i] - Lbulls[i], 0 )  # eq (40)
        
        Ldp_pol, Ld50_pol = P452.dl_p( dc, hc, hts, hrs, hstd, hsrd, ff[i], omega, pp[i], b0, DN )
        
        Ldp[i] = Ldp_pol[int(pol-1)]
        Ld50[i] = Ld50_pol[int(pol-1)]
        
        Lb[i] = P452.bt_loss(ff[i], \
            pp[i], \
            d, \
            h, \
            zone, \
            htg, \
            hrg, \
            phi_path,\
            Gt, \
            Gr, \
            pol, \
            dct, \
            dcr, \
            DN, \
            N0, \
            press, \
            temp, \
            ha_t, \
            ha_r, \
            dk_t, \
            dk_r) 

        
        out1[0,i] = Lbfsg[i] - Lbfsg_ref[i]
        out1str[0] = 'Lbfsg'
        
        out1[1,i] = Lb0p[i] - Lb0p_ref[i]
        out1str[1] = 'Lb0p'        

        out1[2,i] = Lb0b[i] - Lb0b_ref[i]
        out1str[2] = 'Lb0b'
        
        out1[3,i] = Ldsph[i] - Ldsph_ref[i]
        out1str[3] = 'Ldsph'        
        
        out1[4,i] = Ld50[i] - Ld50_ref[i]
        out1str[4] = 'Ld50'
        
        out1[5,i] = Ldp[i] - Ldp_ref[i]
        out1str[5] = 'Ldp'                
        
        out1[6,i] = Lbs[i] - Lbs_ref[i]
        out1str[6] = 'Lbs'        

        out1[7,i] = Lba[i] - Lba_ref[i]
        out1str[7] = 'Lba'        
        
        out1[8,i] = Lb[i] - Lb_ref[i]
        out1str[8] = 'Lb'        
        

# verify error in the results out1 against tolarance
    
    for i in range(0,len(out1str)):
        
        maxi = max(np.abs(out1[i,:]))
       
        (kk,) = np.where(maxi > tol)
        if ~P452.isempty(kk):
            for kki in range(0, len(kk)):
                print('Maximum deviation found in %s larger than tolerance %g:  %g\n' %(out1str[i], tol, maxi) ) 
                failed = True
            
    
    if (not failed):
       success = success + 1
       
    total = total + 1
    
print('Validation results: %d out of %d tests passed successfully.\n' %(success, total))
if (success == total):
    print('The deviation from the reference results is smaller than %g.\n' %(tol))       
    