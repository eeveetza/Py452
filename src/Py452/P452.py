# -*- coding: utf-8 -*-
"""
Created on Thu 17 Mar 2022

@author: eeveetza
"""

import warnings

import numpy as np



def bt_loss(f, p, d, h, zone, htg, hrg, phi_path, Gt, Gr, pol, dct, dcr, DN, N0, press, temp, *args):
    """ 
    P452.bt_loss basic transmission loss according to P.452-17
    Lb = P452.bt_loss(f, p, d, h, zone, htg, hrg, phi_t, phi_r, Gt, Gr, pol, dct, dcr, DN, N0, press, temp, ha_t, ha_r, dk_t, dk_r )

    This is the MAIN function that computes the basic transmission loss not exceeded for p percentage of time
    as defined in ITU-R P.452-17 (Section 4.6). 

    Input parameters:
    f       -   Frequency (GHz)
    p       -   Required time percentage for which the calculated basic
                transmission loss is not exceeded
    d       -   vector of distances di of the i-th profile point (km)
    h       -   vector of heights hi of the i-th profile point (meters
                above mean sea level. Both vectors contain n+1 profile points
    zone    -   Zone type: Coastal land (1), Inland (2) or Sea (3)
    htg     -   Tx Antenna center heigth above ground level (m)
    hrg     -   Rx Antenna center heigth above ground level (m)
    phi_path-   Latitude of path centre between Tx and Rx stations (degrees)
    Gt, Gr  -   Antenna gain in the direction of the horizon along the
                great-circle interference path (dBi)
    pol     -   polarization of the signal (1) horizontal, (2) vertical
    dct     -   Distance over land from the transmit and receive
    dcr         antennas to the coast along the great-circle interference path (km).
                Set to zero for a terminal on a ship or sea platform
    DN      -   The average radio-refractive index lapse-rate through the
                lowest 1 km of the atmosphere (it is a positive quantity in this
                procedure) (N-units/km)
    N0      -   The sea-level surface refractivity, is used only by the
                troposcatter model as a measure of location variability of the
                troposcatter mechanism. The correct values of DN and N0 are given by
                the path-centre values as derived from the appropriate
                maps (N-units)
    press   -   Dry air pressure (hPa)
    temp    -   Air temperature (degrees C)
    ha_t    -   Clutter nominal height (m) at the Tx side
    ha_r    -   Clutter nominal height (m) at the Rx side
    dk_t    -   Clutter nominal distance (km) at the Tx side
    dk_r    -   Clutter nominal distance (km) at the Rx side

    Output parameters:
    Lb     -   basic  transmission loss according to P.452-17

    Example:
    Lb = P452.bt_loss(f, p, d, h, zone, htg, hrg, phi_path, Gt, Gr, pol, dct, dcr, DN, N0, press, temp)
    Lb = P452.bt_loss(f, p, d, h, zone, htg, hrg, phi_path, Gt, Gr, pol, dct, dcr, DN, N0, press, temp, ha_t, ha_r, dk_t, dk_r)

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v0    17MAR22     Ivica Stevanovic, OFCOM         Initial version

    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
    OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
    ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
    OTHER DEALINGS IN THE SOFTWARE.

    THE AUTHOR(S) AND OFCOM (CH) DO NOT PROVIDE ANY SUPPORT FOR THIS SOFTWARE

    """

    # Read the input arguments 
    
    if len(args) > 4:
        print('P452.bt_loss: Too many input arguments; The function requires at most 22')
        print('input arguments. Additional values ignored. Input values may be wrongly assigned.')
    

    # Optional arguments
    
    ha_t = []
    ha_r = []
    dk_t = []
    dk_r = []

    

    icount = 18
    nargin = icount + len(args)
    if nargin >= icount + 1:
        ha_t = args[0]
        if nargin >= icount + 2:
            ha_r = args[1]
            if nargin >= icount + 3:
                dk_t = args[2]
                if nargin >= icount + 4:
                    dk_r = args[3]
    

    # Ensure that vector d is ascending
    if (not issorted(d)):
        raise ValueError('The array of path profile points d(i) must be in ascending order.')
        

    # Ensure that d[0] = 0 (Tx position)
    if d[0] > 0.0 :
        raise ValueError('The first path profile point d[0] = ' + str(d[0]) +  ' must be zero.')

    # Compute the path profile parameters

    # Compute  dtm     -   the longest continuous land (inland + coastal) section of the great-circle path (km)
    zone_r = 12
    dtm = longest_cont_dist(d, zone, zone_r)

    # Compute  dlm     -   the longest continuous inland section of the great-circle path (km)
    zone_r = 2
    dlm = longest_cont_dist(d, zone, zone_r)

    # Compute b0
    b0 = beta0(phi_path, dtm, dlm)

    ae, ab = earth_rad_eff(DN)

    # Compute the path fraction over see

    omega = path_fraction(d, zone, 3)

    # Modify the path according to Section 4.5.4, Step 1 and compute clutter losses
    # only if not isempty ha_t and ha_r

    dc, hc, zonec, htgc, hrgc, Aht, Ahr = closs_corr(f, d, h, zone, htg, hrg, ha_t, ha_r, dk_t, dk_r)

    d = dc
    h = hc
    zone = zonec
    htg = htgc
    hrg = hrgc

    hst, hsr, hstd, hsrd, hte, hre, hm, dlt, dlr, theta_t, theta_r, theta, pathtype = smooth_earth_heights(d, h, htg, hrg, ae, f)

    dtot = d[-1] - d[0]

    #Tx and Rx antenna heights above mean sea level amsl (m)
    hts = h[0] + htgc
    hrs = h[-1] + hrgc

    # Effective Earth curvature Ce (km^-1)

    Ce = 1.0 / ae


    # Find the intermediate profile point with the highest slope of the line
    # from the transmitter to the point

    if len(d) < 4:
        raise ValueError("P452.bt_loss: path profile requires at least 4 points.")
    
    di = d[1:-1]
    hi = h[1:-1]

    Stim = max((hi + 500*Ce*di * (dtot - di) - hts) / di )           # Eq (14)

    # Calculate the slope of the line from transmitter to receiver assuming a
    # LoS path

    Str = (hrs - hts) / dtot                                         # Eq (15)

    # Calculate an interpolation factor Fj to take account of the path angular
    # distance (58)

    THETA = 0.3
    KSI = 0.8

     
    Fj = 1.0 - 0.5*( 1.0 + np.tanh(3.0 * KSI * (Stim-Str)/THETA) )

    # Calculate an interpolation factor, Fk, to take account of the great
    # circle path distance:

    dsw = 20
    kappa = 0.5

    Fk = 1.0 - 0.5*( 1.0 + np.tanh(3.0 * kappa * (dtot-dsw)/dsw) )  # eq (59)
 
    d3D = np.sqrt(dtot*dtot + ((hts-hrs)/1000.0)**2)

    Lbfsg, Lb0p, Lb0b = pl_los(d3D, f, p, b0, omega, temp, press, dlt, dlr)


    Ldp, Ld50 = dl_p( d, h, hts, hrs, hstd, hsrd, f, omega, p, b0, DN )

    # The median basic transmission loss associated with diffraction Eq (43)

    Lbd50 = Lbfsg + Ld50

    # The basic tranmission loss associated with diffraction not exceeded for
    # p% time Eq (44)

    Lbd = Lb0p + Ldp

    # A notional minimum basic transmission loss associated with LoS
    # propagation and over-sea sub-path diffraction

    Lminb0p = Lb0p + (1-omega)*Ldp

    if p >= b0:
        
        Fi = inv_cum_norm(p/100.0)/inv_cum_norm(b0/100.0)      #eq (41a)
    
        Lminb0p = Lbd50 + (Lb0b + (1-omega)*Ldp - Lbd50)*Fi    # eq (60)
    
    

    # Calculate a notional minimum basic transmission loss associated with LoS
    # and transhorizon signal enhancements

    eta = 2.5

    Lba = tl_anomalous(dtot, dlt, dlr, dct, dcr, dlm, hts, hrs, hte, hre, hm, theta_t, theta_r, f, p, temp, press, omega, ae, b0)

    Lminbap = eta*np.log(np.exp(Lba/eta) + np.exp(Lb0p/eta))    # eq (61)

    # Calculate a notional basic transmission loss associated with diffraction
    # and LoS or ducting/layer reflection enhancements

    Lbda = Lbd
 
    if Lminbap <= Lbd[0] :
        Lbda[0] = Lminbap + (Lbd[0]-Lminbap)*Fk
    
    if Lminbap <= Lbd[1] :
        Lbda[1] = Lminbap + (Lbd[1]-Lminbap)*Fk

    # Calculate a modified basic transmission loss, which takes diffraction and
    # LoS or ducting/layer-reflection enhancements into account

    Lbam = Lbda + (Lminb0p - Lbda)*Fj   # eq (63)

    # Calculate the basic transmission loss due to troposcatter not exceeded
    # for any time percantage p 

    Lbs = tl_tropo(dtot, theta, f, p, temp, press, N0, Gt, Gr )

    # Calculate the final transmission loss not exceeded for p% time

    Lb_pol = -5*np.log10(10**(-0.2*Lbs) + 10**(-0.2*Lbam)) + Aht + Ahr  # eq (64)

    Lb = Lb_pol[int(pol-1)]

    return Lb


def tl_tropo(dtot, theta, f, p, temp, press, N0, Gt, Gr ):
    """
    tl_tropo Basic transmission loss due to troposcatterer to P.452-17
    Lbs = tl_tropo(dtot, theta, f, p, temp, press, N0, Gt, Gr )

    This function computes the basic transmission loss due to troposcatterer 
    not exceeded for p of time
    as defined in ITU-R P.452-17 (Section 4.3)

        Input parameters:
        dtot    -   Great-circle path distance (km)
        theta   -   Path angular distance (mrad)
        f       -   frequency expressed in GHz
        p       -   percentage of time
        temp    -   Temperature (deg C)
        press   -   Dry air pressure (hPa)
        N0      -   path centre sea-level surface refractivity derived from Fig. 6
        Gt,Gr   -   Antenna gain in the direction of the horizon along the
                    great-circle interference path (dBi)

        Output parameters:
        Lbs    -   the basic transmission loss due to troposcatterer 
                    not exceeded for p% of time

        Example:
        Lbs = tl_tropo(dtot, theta, f, p, temp, press, N0, Gt, Gr )
        

        Rev   Date        Author                          Description
        -------------------------------------------------------------------------------
        v0    17MAR22    Ivica Stevanovic, OFCOM         Initial version
    """

    T = temp + 273.15

    # Frequency dependent loss

    Lf = 25*np.log10(f) - 2.5*(np.log10(f/2.0))**2   # eq (45a)

    # aperture to medium coupling loss (dB)

    Lc = 0.051*np.exp(0.055*(Gt+Gr))             # eq (45b)

    # gaseous absorbtion derived from equation (9) using rho = 3 g/m^3 for the
    # whole path length

    rho = 3

    # compute specific attenuation due to dry air and water vapor:
    g_0, g_w = p676d11_ga(f, press, rho, T)

    Ag = (g_0 + g_w) * dtot  #(9)

    # the basic transmission loss due to troposcatter not exceeded for any time
    # percentage p, below 50# is given

    Lbs = 190 + Lf + 20*np.log10(dtot) + 0.573*theta - 0.15*N0 + Lc + Ag - 10.1*(-np.log10(p/50.0))**(0.7)

    return Lbs


def p676d11_ga(f, p, rho, T):
    """
    p676d11_ga Specific attenuation due to dry air and water vapour
    g_0, g_w = p676d11_ga(f, p, rho, T)
    This function computes the specific attenuation due to dry air and water vapour,
    at frequencies up to 1 000 GHz for different values of of pressure, temperature
    and humidity by means of a summation of the individual resonance lines from
    oxygen and water vapour according to ITU-R P.676-11

    Input parameters:
    f       -   Frequency (GHz)
    p       -   Dry air pressure (hPa)
    rho     -   Water vapor density (g/m^3)
    T       -   Temperature (K)

    Output parameters:
    g_o, g_w   -   specific attenuation due to dry air and water vapour


    Rev   Date        Author                          Description
    ----------------------------------------------------------------------------------------------
    v0    17MAR22     Ivica Stevanovic, OFCOM         First implementation of P.676-11
    """

    ## spectroscopic data for oxigen
    #             f0        a1    a2     a3   a4     a5     a6
    oxigen = np.mat( 
        [[50.474214, 0.975, 9.651, 6.690, 0.0, 2.566, 6.850],
        [50.987745, 2.529, 8.653, 7.170, 0.0, 2.246, 6.800],
        [51.503360, 6.193, 7.709, 7.640, 0.0, 1.947, 6.729],
        [52.021429, 14.320, 6.819, 8.110, 0.0, 1.667, 6.640],
        [52.542418, 31.240, 5.983, 8.580, 0.0, 1.388, 6.526],
        [53.066934, 64.290, 5.201, 9.060, 0.0, 1.349, 6.206],
        [53.595775, 124.600, 4.474, 9.550, 0.0, 2.227, 5.085],
        [54.130025, 227.300, 3.800, 9.960, 0.0, 3.170, 3.750],
        [54.671180, 389.700, 3.182, 10.370, 0.0, 3.558, 2.654],
        [55.221384, 627.100, 2.618, 10.890, 0.0, 2.560, 2.952],
        [55.783815, 945.300, 2.109, 11.340, 0.0, -1.172, 6.135],
        [56.264774, 543.400, 0.014, 17.030, 0.0, 3.525, -0.978],
        [56.363399, 1331.800, 1.654, 11.890, 0.0, -2.378, 6.547],
        [56.968211, 1746.600, 1.255, 12.230, 0.0, -3.545, 6.451],
        [57.612486, 2120.100, 0.910, 12.620, 0.0, -5.416, 6.056],
        [58.323877, 2363.700, 0.621, 12.950, 0.0, -1.932, 0.436],
        [58.446588, 1442.100, 0.083, 14.910, 0.0, 6.768, -1.273],
        [59.164204, 2379.900, 0.387, 13.530, 0.0, -6.561, 2.309],
        [59.590983, 2090.700, 0.207, 14.080, 0.0, 6.957, -0.776],
        [60.306056, 2103.400, 0.207, 14.150, 0.0, -6.395, 0.699],
        [60.434778, 2438.000, 0.386, 13.390, 0.0, 6.342, -2.825],
        [61.150562, 2479.500, 0.621, 12.920, 0.0, 1.014, -0.584],
        [61.800158, 2275.900, 0.910, 12.630, 0.0, 5.014, -6.619],
        [62.411220, 1915.400, 1.255, 12.170, 0.0, 3.029, -6.759],
        [62.486253, 1503.000, 0.083, 15.130, 0.0, -4.499, 0.844],
        [62.997984, 1490.200, 1.654, 11.740, 0.0, 1.856, -6.675],
        [63.568526, 1078.000, 2.108, 11.340, 0.0, 0.658, -6.139],
        [64.127775, 728.700, 2.617, 10.880, 0.0, -3.036, -2.895],
        [64.678910, 461.300, 3.181, 10.380, 0.0, -3.968, -2.590],
        [65.224078, 274.000, 3.800, 9.960, 0.0, -3.528, -3.680],
        [65.764779, 153.000, 4.473, 9.550, 0.0, -2.548, -5.002],
        [66.302096, 80.400, 5.200, 9.060, 0.0, -1.660, -6.091],
        [66.836834, 39.800, 5.982, 8.580, 0.0, -1.680, -6.393],
        [67.369601, 18.560, 6.818, 8.110, 0.0, -1.956, -6.475],
        [67.900868, 8.172, 7.708, 7.640, 0.0, -2.216, -6.545],
        [68.431006, 3.397, 8.652, 7.170, 0.0, -2.492, -6.600],
        [68.960312, 1.334, 9.650, 6.690, 0.0, -2.773, -6.650],
        [118.750334, 940.300, 0.010, 16.640, 0.0, -0.439, 0.079],
        [368.498246, 67.400, 0.048, 16.400, 0.0, 0.000, 0.000],
        [424.763020, 637.700, 0.044, 16.400, 0.0, 0.000, 0.000],
        [487.249273, 237.400, 0.049, 16.000, 0.0, 0.000, 0.000],
        [715.392902, 98.100, 0.145, 16.000, 0.0, 0.000, 0.000],
        [773.839490, 572.300, 0.141, 16.200, 0.0, 0.000, 0.000],
        [834.145546, 183.100, 0.145, 14.700, 0.0, 0.000, 0.000]] 
    )

    ## spectroscopic data for water-vapor #Table 2, modified in version P.676-11
    #            f0       b1    b2    b3   b4   b5   b6
    vapor = np.mat(
        [[22.235080, .1079, 2.144, 26.38, .76, 5.087, 1.00],
        [67.803960,  .0011, 8.732, 28.58, .69, 4.930, .82],
        [119.995940, .0007, 8.353, 29.48, .70, 4.780, .79],
        [183.310087, 2.273, 0.668, 29.06, .77, 5.022, .85],
        [321.225630, .0470, 6.179, 24.04, .67, 4.398, .54],
        [325.152888, 1.514, 1.541, 28.23, .64, 4.893, .74],
        [336.227764, .0010, 9.825, 26.93, .69, 4.740, .61],
        [380.197353, 11.67, 1.048, 28.11, .54, 5.063, .89],
        [390.134508, .0045, 7.347, 21.52, .63, 4.810, .55],
        [437.346667, .0632, 5.048, 18.45, .60, 4.230, .48],
        [439.150807, .9098, 3.595, 20.07, .63, 4.483, .52],
        [443.018343, .1920, 5.048, 15.55, .60, 5.083, .50],
        [448.001085, 10.41, 1.405, 25.64, .66, 5.028, .67],
        [470.888999, .3254, 3.597, 21.34, .66, 4.506, .65],
        [474.689092, 1.260, 2.379, 23.20, .65, 4.804, .64],
        [488.490108, .2529, 2.852, 25.86, .69, 5.201, .72],
        [503.568532, .0372, 6.731, 16.12, .61, 3.980, .43],
        [504.482692, .0124, 6.731, 16.12, .61, 4.010, .45],
        [547.676440, .9785, .158, 26.00, .70, 4.500, 1.00],
        [552.020960, .1840, .158, 26.00, .70, 4.500, 1.00],
        [556.935985, 497.0, .159, 30.86, .69, 4.552, 1.00],
        [620.700807, 5.015, 2.391, 24.38, .71, 4.856, .68],
        [645.766085, .0067, 8.633, 18.00, .60, 4.000, .50],
        [658.005280, .2732, 7.816, 32.10, .69, 4.140, 1.00],
        [752.033113, 243.4, .396, 30.86, .68, 4.352, .84],
        [841.051732, .0134, 8.177, 15.90, .33, 5.760, .45],
        [859.965698, .1325, 8.055, 30.60, .68, 4.090, .84],
        [899.303175, .0547, 7.914, 29.85, .68, 4.530, .90],
        [902.611085, .0386, 8.429, 28.65, .70, 5.100, .95],
        [906.205957, .1836, 5.110, 24.08, .70, 4.700, .53],
        [916.171582, 8.400, 1.441, 26.73, .70, 5.150, .78],
        [923.112692, .0079, 10.293, 29.00, .70, 5.000, .80],
        [970.315022, 9.009, 1.919, 25.50, .64, 4.940, .67],
        [987.926764, 134.6, .257, 29.85, .68, 4.550, .90],
        [1780.000000, 17506., .952, 196.3, 2.00, 24.15, 5.00]]
    )

    a1 = np.squeeze(np.asarray(oxigen[:,1]))
    a2 = np.squeeze(np.asarray(oxigen[:,2]))
    a3 = np.squeeze(np.asarray(oxigen[:,3]))
    a4 = np.squeeze(np.asarray(oxigen[:,4]))
    a5 = np.squeeze(np.asarray(oxigen[:,5]))
    a6 = np.squeeze(np.asarray(oxigen[:,6]))
    b1 = np.squeeze(np.asarray(vapor[:,1]))
    b2 = np.squeeze(np.asarray(vapor[:,2]))
    b3 = np.squeeze(np.asarray(vapor[:,3]))
    b4 = np.squeeze(np.asarray(vapor[:,4]))
    b5 = np.squeeze(np.asarray(vapor[:,5]))
    b6 = np.squeeze(np.asarray(vapor[:,6]))

    theta = 300.0/T

    e = rho * T / 216.7        # equation (4)

    ## Oxigen computation
    fi = np.squeeze(np.asarray(oxigen[:,0]))

    Si = a1 * 1e-7 * p * theta**3 * np.exp(a2 * (1.0 - theta))       # equation (3)

    df = a3 * 1e-4 * (p * theta **(0.8-a4) + 1.1 * e * theta)  # equation (6a)

    # Doppler broadening

    df = np.sqrt( df * df + 2.25e-6)                                   # equation (6b)

    delta = (a5 + a6 * theta) * 1e-4 * (p + e) * theta**(0.8)     # equation (7)

    Fi = f / fi * (  (df - delta * (fi - f)) / ( (fi - f)**2 + df ** 2  ) + \
        (df - delta * (fi + f))  / ( (fi + f)**2 + df**2  ))        # equation (5)

    d = 5.6e-4 * (p + e) * theta ** (0.8)                            # equation (9)

    Ndf = f * p * theta**2 * ( 6.14e-5/(d * (1 + (f/d) ** 2) ) + \
        1.4e-12 * p * theta ** (1.5)/(1 + 1.9e-5 * f ** (1.5)) )       # equation (8)

    # specific attenuation due to dry air (oxygen, pressure induced nitrogen
    # and non-resonant Debye attenuation), equations (1-2)

    g_0 = 0.182 * f * (np.dot(Si, Fi) + Ndf)


    ## vapor computation

    fi = np.squeeze(np.asarray(vapor[:,0]))

    Si = b1 * 1e-1 * e * theta ** 3.5 * np.exp(b2 * (1.0 - theta))      # equation (3)

    df = b3 * 1e-4 * (p * theta ** (b4) + b5 * e * theta ** b6)     # equation (6a)

    # Doppler broadening

    df = 0.535 * df + np.sqrt( 0.217* df * df + 2.1316e-12 * fi * fi/theta) # equation (6b)

    delta = 0                                                           # equation (7)

    Fi = f / fi * (  (df - delta * (fi - f)) / ( (fi - f) ** 2 + df ** 2  ) + \
        (df - delta * (fi + f)) / ( (fi + f)**2 + df**2  ))              # equation (5)

    # specific attenuation due to water vapour, equations (1-2)

    g_w = 0.182 * f * (np.dot(Si, Fi) )

    return g_0, g_w
    
def tl_anomalous(dtot, dlt, dlr, dct, dcr, dlm, hts, hrs, hte, hre, hm, theta_t, theta_r, f, p, temp, press, omega, ae, b0):
    """
    tl_anomalous Basic transmission loss due to anomalous propagation according to P.452-17
    Lba = tl_anomalous(dtot, dlt, dlr, dct, dcr, dlm, hts, hrs, hte, hre, hm, theta_t, theta_r, f, p, temp, press, omega, ae, b0)

    This function computes the basic transmission loss occuring during
    periods of anomalous propagation (ducting and layer reflection)
    as defined in ITU-R P.452-17 (Section 4.4)

        Input parameters:
        dtot         -   Great-circle path distance (km)
        dlt          -   interfering antenna horizon distance (km)
        dlr          -   Interfered-with antenna horizon distance (km)
        dct, dcr     -   Distance over land from the transmit and receive
                        antennas tothe coast along the great-circle interference path (km).
                        Set to zero for a terminal on a ship or sea platform
        dlm          -   the longest continuous inland section of the great-circle path (km)
        hts, hrs     -   Tx and Rx antenna heights aobe mean sea level amsl (m)
        hte, hre     -   Tx and Rx terminal effective heights for the ducting/layer reflection model (m)
        hm           -   The terrain roughness parameter (m)
        theta_t      -   Interfering antenna horizon elevation angle (mrad)
        theta_r      -   Interfered-with antenna horizon elevation angle (mrad)
        f            -   frequency expressed in GHz
        p            -   percentage of time
        temp         -   Temperature (deg C)
        press        -   Dry air pressure (hPa)
        omega        -   fraction of the total path over water
        ae           -   the median effective Earth radius (km)
        b0           -   the time percentage that the refractivity gradient (DELTA-N) exceeds 100 N-units/km in the first 100m of the lower atmosphere

        Output parameters:
        Lba    -   the basic transmission loss due to anomalous propagation
                (ducting and layer reflection)

        Example:
        Lba = tl_anomalous(dtot, dlt, dlr, dct, dcr, dlm, hts, hrs, hte, hre, hm, theta_t, theta_r, f, p, temp, press, omega, b0)
        

        Rev   Date        Author                          Description
        -------------------------------------------------------------------------------
        v0    22MAR22     Ivica Stevanovic, OFCOM         Initial version


    """
    
    ## Body of function

    # empirical correction to account for the increasing attenuation with
    # wavelength inducted propagation (47a)

    Alf = 0

    if f < 0.5:
        Alf = 45.375 - 137.0*f + 92.5*f*f
    

    # site-shielding diffraction losses for the interfering and interfered-with
    # stations (48)

    theta_t1 = theta_t - 0.1*dlt    # eq (48a)
    theta_r1 = theta_r - 0.1*dlr

    Ast = 0
    Asr = 0

    if theta_t1 > 0:
        Ast = 20*np.log10(1 + 0.361*theta_t1*np.sqrt(f*dlt)) + 0.264*theta_t1*f**(1.0/3.0)
    

    if theta_r1 > 0:
        Asr = 20*np.log10(1 + 0.361*theta_r1*np.sqrt(f*dlr)) + 0.264*theta_r1*f**(1.0/3.0)
    

    # over-sea surface duct coupling correction for the interfering and
    # interfered-with stations (49) and (49a)

    Act = 0
    Acr = 0

    if dct <= 5:
        if dct <= dlt:
            if omega >= 0.75:
                Act = -3*np.exp(-0.25*dct*dct)*(1+ np.tanh( 0.07*(50-hts) ))
            
        
    

    if dcr <= 5:
        if dcr <= dlr:
            if omega >= 0.75:
                Acr = -3*np.exp(-0.25*dcr*dcr)*(1+ np.tanh( 0.07*(50-hrs) ))
            
        
    

    # specific attenuation (51)

    gamma_d = 5e-5 * ae * f ** (1.0/3.0)

    # angular distance (corrected where appropriate) (52-52a)

    theta_t1 = theta_t
    theta_r1 = theta_r

    if theta_t> 0.1*dlt:
        theta_t1 = 0.1*dlt
    

    if theta_r > 0.1*dlr:
        theta_r1 = 0.1*dlr
    

    theta1 = 1e3*dtot/ae + theta_t1 + theta_r1   

    dI = min(dtot - dlt - dlr, 40)   # eq (56a)

    mu3 = 1

    if hm > 10:
        
        mu3 = np.exp( -4.6e-5 * (hm-10)*(43+6*dI) )  # eq (56)

        
    

    tau = 1 - np.exp(-(4.12e-4*dlm**2.41))       # eq (3a)

    epsilon = 3.5

    alpha = -0.6 - epsilon*1e-9*dtot**(3.1)*tau   # eq (55a)

    if alpha < -3.4:
        alpha = -3.4
    
    # correction for path geometry:

    mu2 = ( 500/ae * dtot**2/( np.sqrt(hte) + np.sqrt(hre) )**2 )**alpha

    if mu2 > 1:
        mu2 = 1
    

    beta = b0 * mu2 * mu3      # eq (54)

    #beta = max(beta, eps)      # to avoid division by zero

    Gamma = 1.076/(2.0058-np.log10(beta))**1.012 * \
            np.exp( -( 9.51 - 4.8*np.log10(beta) + 0.198*(np.log10(beta)) ** 2)*1e-6*dtot**(1.13) )

    # time percentage variablity (cumulative distribution):

    Ap = -12 + (1.2 + 3.7e-3*dtot)*np.log10(p/beta) + 12 * (p/beta)**Gamma  # eq (53)

    # time percentage and angular-distance dependent losses within the
    # anomalous propagation mechanism

    Adp = gamma_d*theta1 + Ap   # eq (50)

    # gaseous absorbtion derived from equation (9) using rho = 3 g/m^3 for the
    # whole path length

    # water vapor density

    rho = 7.5 + 2.5*omega

    T = temp + 273.15

    # compute specific attenuation due to dry air and water vapor:
    g_0, g_w = p676d11_ga(f, press, rho, T)

    Ag = (g_0 + g_w) * dtot  #(9)

    # total of fixed coupling losses (except for local clutter losses) between
    # the antennas and the anomalous propagation structure within the
    # atmosphere (47)

    Af = 102.45 + 20*np.log10(f) + 20*np.log10(dlt + dlr) + Alf + Ast + Asr + Act + Acr

    # total basic transmission loss occuring during periods of anomalaous
    # propagation 

    Lba = Af + Adp + Ag

    return Lba
    

def  smooth_earth_heights(d, h, htg, hrg, ae, f):
    """
    smooth_earth_heights smooth-Earth effective antenna heights according to ITU-R P.452-17
    [hst, hsr, hstd, hsrd, hte, hre, hm, dlt, dlr, theta_t, theta_r, theta_tot, pathtype] = smooth_earth_heights(d, h, htg, hrg, ae, f)
    This function derives smooth-Earth effective antenna heights according to
    Sections 4 and 5 of the Annex 2 of ITU-R P.452-17

    Input parameters:
    d         -   vector of terrain profile distances from Tx [0,dtot] (km)
    h         -   vector of terrain profile heigths amsl (m)
    htg, hrg  -   Tx and Rx antenna heights above ground level (m)
    ae        -   median effective Earth's radius (c.f. Eq (6a))
    f         -   frequency (GHz)

    Output parameters:

    hst, hsr     -   Tx and Rx antenna heigts of the smooth-Earth surface amsl (m)
    hstd, hsrd   -   Tx and Rx effective antenna heigts for the diffraction model (m)
    hte, hre     -   Tx and Rx terminal effective heights for the ducting/layer reflection model (m)
    hm           -   The terrain roughness parameter (m)
    dlt          -   interfering antenna horizon distance (km)
    dlr          -   Interfered-with antenna horizon distance (km)
    theta_t      -   Interfering antenna horizon elevation angle (mrad)
    theta_r      -   Interfered-with antenna horizon elevation angle (mrad)
    theta_tot    -   Angular distance (mrad)
    pathtype     -   1 = 'los', 2 = 'transhorizon'

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v0    17MAR22     Ivica Stevanovic, OFCOM         First implementation in python
    """
    n = len(d)

    dtot = d[-1]

    #Tx and Rx antenna heights above mean sea level amsl (m)
    hts = h[0] + htg
    hrs = h[-1] + hrg

    # Section 5.1.6.2

    v1 = 0
    for ii in range(1,n):
        v1 = v1 + (d[ii]-d[ii-1])*(h[ii]+h[ii-1])  # Eq (161)
    
    v2 = 0
    for ii in range(1,n):
        v2 = v2 + (d[ii]-d[ii-1])*( h[ii]*( 2*d[ii] + d[ii-1] ) + h[ii-1] * ( d[ii] + 2*d[ii-1] ) )  # Eq (162)
    

    hst = (2*v1*dtot - v2)/dtot ** 2       # Eq (163)
    hsr = (v2- v1*dtot)/dtot ** 2          # Eq (164)

    # Section 5.1.6.3

    HH = h - (hts*(dtot-d) + hrs*d)/dtot  # Eq (165d)

    hobs = max(HH[1:-1])                 # Eq (165a)

    alpha_obt = max( HH[1:-1] / d[1:-1] ) # Eq (165b)

    alpha_obr = max( HH[1:-1] / ( dtot - d[1:-1] ) ) # Eq (165c)

    # Calculate provisional values for the Tx and Rx smooth surface heights

    gt = alpha_obt/(alpha_obt + alpha_obr)         # Eq (166e)
    gr = alpha_obr/(alpha_obt + alpha_obr)         # Eq (166f)

    if hobs <= 0:
        hstp = hst                                 # Eq (166a)
        hsrp = hsr                                 # Eq (166b)
    else:
        hstp = hst - hobs*gt                       # Eq (166c)
        hsrp = hsr - hobs*gr                       # Eq (166d)
    

    # calculate the final values as required by the diffraction model

    if hstp >= h[0]:
        hstd = h[0]                                # Eq (167a)
    else:
        hstd = hstp                                # Eq (167b)
    

    if hsrp > h[-1]:
        hsrd = h[-1]                              # Eq (167c)
    else:
        hsrd = hsrp                                # Eq (167d)
    

    # Interfering antenna horizon elevation angle and distance

    ii = range(1,n-1)

    theta = 1000 * np.arctan( (h[ii] - hts) / (1000 * d[ii] ) - d[ii] / (2*ae) )  # Eq (152)

    #theta(theta < 0) = 0  # condition below equation (152)

    theta_td = 1000 * np.arctan( (hrs - hts) / (1000 * dtot ) - dtot / (2*ae) )  # Eq (153)
    theta_rd = 1000 * np.arctan( (hts - hrs) / (1000 * dtot ) - dtot / (2*ae) )  # Eq (156a)


    #theta_t = max(theta)                           # Eq (154)
    theta_max = max(theta)
    theta_t = max(theta_max, theta_td)                           # Eq (154)


    if theta_t > theta_td:   # Eq (150): test for the trans-horizon path
        pathtype = 2 #transhorizon
    else:
        pathtype = 1 #los
    


    (kindex, ) = np.where(theta == theta_max)

    lt = kindex[0]+1

    dlt = d[lt]                             # Eq (155)

    # Interfered-with antenna horizon elevation angle and distance

    theta = 1000 * np.arctan( (h[ii] - hrs) / (1000 * (dtot - d[ii]) ) - (dtot - d[ii]) / (2*ae) )  # Eq (157)

    #theta(theta < 0) = 0

    #theta_r = max(theta)
    theta_max = max(theta)
    theta_r = max(theta_max, theta_rd)

    (kindex,) = np.where(theta == theta_max)
    lr = kindex[-1] + 1

    dlr = dtot - d[lr]                            # Eq (158)

    if pathtype == 1:
    
        theta_t = theta_td
        theta_r = theta_rd
        
        ii = range(1, n-1)
        
        lam = 0.2998/f
        Ce = 1/ae
        
        nu = (h[ii] + 500*Ce*d[ii] * (dtot-d[ii])- (hts*(dtot - d[ii]) + hrs * d[ii])/dtot) * \
            np.sqrt(0.002*dtot / (lam * d[ii] * (dtot-d[ii])))
        numax = max(nu)
        
        (kindex,) = np.where(nu == numax)
        lt = kindex[-1] + 1  
        dlt = d[lt]  
        dlr = dtot - dlt
        (kindex,) = np.where(dlr <=dtot - d[ii])
        lr = kindex[-1] + 1
    

    # Angular distance

    theta_tot = 1e3 * dtot/ae + theta_t + theta_r         # Eq (159)


    # Section 5.1.6.4 Ducting/layer-reflection model

    # Calculate the smooth-Earth heights at transmitter and receiver as
    # required for the roughness factor

    hst = min(hst, h[0])                           # Eq (168a)
    hsr = min(hsr, h[-1])                         # Eq (168b)

    # Slope of the smooth-Earth surface

    m = (hsr - hst)/ dtot                          # Eq (169)

    # The terminal effective heigts for the ducting/layer-reflection model

    hte = htg + h[0] -   hst                       # Eq (170)
    hre = hrg + h[-1] - hsr                       

    ii = range(lt, lr + 1)

    hm = max(h[ii] - (hst + m*d[ii] ))

    return hst, hsr, hstd, hsrd, hte, hre, hm, dlt, dlr, theta_t, theta_r, theta_tot, pathtype


def pl_los(d, f, p, b0, w, temp, press, dlt, dlr):
    """
    pl_los Line-of-sight transmission loss according to ITU-R P.452-17
    This function computes line-of-sight transmission loss (including short-term effects)
    as defined in ITU-R P.452-17.

    Input parameters:
    d       -   Great-circle path distance (km)
    f       -   Frequency (GHz)
    p       -   Required time percentage(s) for which the calculated basic
                transmission loss is not exceeded (%)
    b0      -   Point incidence of anomalous propagation for the path
                central location (%)
    w       -   Fraction of the total path over water (%)
    temp    -   Temperature (degrees C)
    press   -   Dry air pressure (hPa)
    dlt     -   For a transhorizon path, distance from the transmit antenna to
                its horizon (km). For a LoS path, each is set to the distance
                from the terminal to the profile point identified as the Bullington
                point in the diffraction method for 50% time
    dlr     -   For a transhorizon path, distance from the receive antenna to
                its horizon (km). The same note as for dlt applies here.

    Output parameters:
    Lbfsg   -   Basic transmission loss due to free-space propagation and
                attenuation by atmospheric gases
    Lb0p    -   Basic transmission loss not exceeded for time percentage, p%, due to LoS propagation
    Lb0b    -   Basic transmission loss not exceedd for time percentage, b0%, due to LoS propagation

    Example:
    [Lbfsg, Lb0p, Lb0b] = pl_los(d, f, p, b0, w, temp, press, dlt, dlr)

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v0    18MAR22     Ivica Stevanovic, OFCOM         First implementation

    """
    
    T = temp + 273.15

    # water vapor density
    rho = 7.5 + 2.5 * w  # (9a)

    # compute specific attenuation due to dry air and water vapor:
    g_0, g_w = p676d11_ga(f, press, rho, T)

    Ag = (g_0 + g_w) * d  #(9)

    # Basic transmission loss due to free-space propagation and attenuation
    # by atmospheric gases
    
    Lbfsg = 92.4 + 20.0*np.log10(f) + 20.0*np.log10(d) + Ag  # (8)

    # Corrections for multipath and focusing effects at p and b0
    Esp = 2.6 * (1 - np.exp(-0.1 * (dlt + dlr) ) ) * np.log10(p/50)   #(10a)
    Esb = 2.6 * (1 - np.exp(-0.1 * (dlt + dlr) ) ) * np.log10(b0/50)  #(10b)

    # Basic transmission loss not exceeded for time percentage p# due to
    # LoS propagation
    Lb0p = Lbfsg + Esp    #(11)

    # Basic transmission loss not exceeded for time percentage b0% due to
    # LoS propagation
    Lb0b = Lbfsg + Esb    #(12)

    return Lbfsg, Lb0p, Lb0b


def  path_fraction(d, zone, zone_r):
    """
    path_fraction Path fraction belonging to a given zone_r
     omega = path_fraction(d, zone, zone_r)
     This function computes the path fraction belonging to a given zone_r
     of the great-circle path (km) 

     Input arguments:
     d       -   vector of distances in the path profile
     zone    -   vector of zones in the path profile
     zone_r  -   reference zone for which the fraction is computed

     Output arguments:
     omega   -   path fraction belonging to the given zone_r

     Example:
     omega = path_fraction(d, zone, zone_r)

     Rev   Date        Author                          Description
     -------------------------------------------------------------------------------
     v0    17MAR22     Ivica Stevanovic, OFCOM         First implementation in python
    """
    dm = 0

    start,stop = find_intervals((zone == zone_r))

    n = len(start)

    for i in range(0,n):
        delta = 0
        if ( d[stop[i]] < d[-1] ):
            delta = delta + ( d[stop[i]+1]-d[stop[i]] )/2.0
        
        
        if ( d[start[i]] > 0 ) :
            delta = delta + ( d[stop[i]]-d[stop[i]-1] )/2.0
        
        dm = dm + d[stop[i]]-d[start[i]] + delta

    omega = dm/(d[-1]-d[0])

    return omega


def longest_cont_dist(d, zone, zone_r):
    """
    longest_cont_dist Longest continuous path belonging to the zone_r
    dm = longest_cont_dist(d, zone, zone_r)
    This function computes the longest continuous section of the
    great-circle path (km) for a given zone_r

    Input arguments:
    d       -   vector of distances in the path profile
    zone    -   vector of zones in the path profile
    zone_r  -   reference zone for which the longest continuous section
                is computed

    Output arguments:
    dm      -   the longest continuous section of the great-circle path (km) for a given zone_r

    Example:
    dm = longest_cont_dist(d, zone, zone_r)

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v0    18MAR22     Ivica Stevanovic, OFCOM         Initial version
         
    """     
    dm = 0

    if zone_r  == 12:
        start,stop = find_intervals((zone == 1)+(zone==2))
    else:
        start,stop = find_intervals((zone == zone_r))
    
    n = len(start)

    for i in range(0,n): 
        delta = 0
        if (d[stop[i]] < d[-1]):
            delta = delta + ( d[stop[i]+1]-d[stop[i]] )/2.0
        
        
        if (d[start[i]]>0):
            delta = delta + ( d[stop[i]]-d[stop[i]-1] )/2.0
        
        
        dm = max(d[stop[i]]-d[start[i]] + delta, dm)
    
    return dm

def inv_cum_norm( x ):
    """
    inv_cum_norm approximation to the inverse cummulative normal distribution
    I = inv_cum_norm( x )
    This function implements an approximation to the inverse cummulative
    normal distribution function for x <= 0.5 as defined in Attachment 3 to
    Annex 1 of the ITU-R P.452-17

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v0    18MAR22     Ivica Stevanovic, OFCOM         Initial version
    """
    if x < 0.000001:
        x = 0.000001


    if x > 0.5:
        warnings.warn('Warning: Function inv_cum_norm is defined for arguments not larger than 0.5')

    tx = np.sqrt(-2.0*np.log(x))    # eq (172a)

    C0 = 2.515516698        # eq (172c)
    C1 = 0.802853           # eq (172d)
    C2 = 0.010328           # eq (172e)
    D1 = 1.432788           # eq (172f)
    D2 = 0.189269           # eq (172g)
    D3 = 0.001308           # eq (172h)

    ksi = ( (C2*tx+C1)*tx + C0 ) / ( ((D3*tx + D2)*tx + D1)*tx + 1 )  # eq (172b)

    I = ksi - tx           # eq (172)

    return I


def great_circle_path(Phire, Phite, Phirn, Phitn, Re, dpnt):
    """
    great_circle_path Great-circle path calculations according to Attachment H
    This function computes the great-circle intermediate points on the
    radio path as defined in ITU-R P.2001-4 Attachment H

        Input parameters:
        Phire   -   Receiver longitude, positive to east (deg)
        Phite   -   Transmitter longitude, positive to east (deg)
        Phirn   -   Receiver latitude, positive to north (deg)
        Phitn   -   Transmitter latitude, positive to north (deg)
        Re      -   Average Earth radius (km)
        dpnt    -   Distance from the transmitter to the intermediate point (km)

        Output parameters:
        Phipnte -   Longitude of the intermediate point (deg)
        Phipntn -   Latitude of the intermediate point (deg)
        Bt2r    -   Bearing of the great-circle path from Tx towards the Rx (deg)
        dgc     -   Great-circle path length (km)

        Example:
        [Bt2r, Phipnte, Phipntn, dgc] = great_circle_path(Phire, Phite, Phirn, Phitn, Re, dpnt)

        Rev   Date        Author                          Description
        -------------------------------------------------------------------------------
        v0    05SEP22     Ivica Stevanovic, OFCOM         Initial version
    """
    ## H.2 Path length and bearing

    # Difference (deg) in longitude between the terminals (H.2.1)

    Dlon = Phire - Phite

    # Calculate quantity r (H.2.2)

    r = sind(Phitn) * sind(Phirn) + cosd(Phitn) * cosd(Phirn) * cosd(Dlon)

    # Calculate the path length as the angle subtended at the center of
    # average-radius Earth (H.2.3)

    Phid = np.arccos(r)  # radians

    # Calculate the great-circle path length (H.2.4)

    dgc = Phid * Re  # km

    # Calculate the quantity x1 (H.2.5a)

    x1 = sind(Phirn)-r*sind(Phitn)

    # Calculate the quantity y1 (H.2.5b)

    y1 = cosd(Phitn)*cosd(Phirn)*sind(Dlon)

    # Calculate the bearing of the great-circle path for Tx to Rx (H.2.6)

    if (abs(x1) < 1e-9 and abs(y1) < 1e-9 ):
        Bt2r = Phire
    else:
        Bt2r = atan2d(y1,x1)
    

    ## H.3 Calculation of intermediate path point

    # Calculate the distance to the point as the angle subtended at the center
    # of average-radius Earth (H.3.1)

    Phipnt = dpnt/Re  #radians

    # Calculate quantity s (H.3.2)

    s = sind(Phitn)*np.cos(Phipnt) + cosd(Phitn)*np.sin(Phipnt)*cosd(Bt2r)

    # The latitude of the intermediate point is now given by (H.3.3)

    Phipntn = asind(s) # degs

    # Calculate the quantity x2 (H.3.4a)

    x2 = np.cos(Phipnt)-s*sind(Phitn)

    # Calculate the quantity y2 (H.3.4b)

    y2 = cosd(Phitn)*np.sin(Phipnt)*sind(Bt2r)

    # Calculate the longitude of the intermediate point Phipnte (H.3.5)

    if (x2 < 1e-9 and y2 < 1e-9):
        Phipnte = Bt2r
    else:
        Phipnte = Phite + atan2d(y2,x2)
    

    return Phipnte, Phipntn, Bt2r, dgc

def isempty(x):
    if np.size(x) == 0:
        return True
    else:
        return False


def issorted(a):
    if (np.all(np.diff(a) >= 0)):
        return True
    else:
        return False

def sind(x):
    return  np.sin(x * np.pi/180.0)

def cosd(x):
    return  np.cos(x * np.pi/180.0)

def asind(x):
    return  np.arcsin(x) * 180.0/np.pi

def atan2d(y, x):
    return  np.arctan2(y, x) * 180.0/np.pi
    

def find_intervals(series):
    """
    find_intervals Find all intervals with consecutive 1's
     [k1, k2] = find_intervals(series)
     This function finds all 1's intervals, namely, the indices when the
     intervals start and where they end

     For example, for the input indices
           0 0 1 1 1 1 0 0 0 1 1 0 0
     this function will give back
       k1 = 3, 10
       k2 = 6, 11

     Input arguments:
     indices -   vector containing zeros and ones

     Output arguments:
     k1      -   vector of start-indices of the found intervals
     k2      -   vector of end-indices of the found intervals

     Example:
     [k1, k2] = find_intervals(indices)

     Rev   Date        Author                          Description
     -------------------------------------------------------------------------------
     v0    18MAR22     Ivica Stevanovic, OFCOM         First implementation in python
     
    """
    k1 = []
    k2 = []
    series_int = 1 * series
    # make sure series is  is a row vector
    
    #if (size(series,1) > 1):
    #    series = series.'
        

    if max(series_int) == 1:
        (k1,) = np.where(np.diff(  np.append(0, series_int)  ) ==  1)
        (k2,) = np.where(np.diff(  np.append(series_int, 0)  ) == -1)
        
    return k1, k2
    
    
def   earth_rad_eff(DN):
    """
    earth_rad_eff Median value of the effective Earth radius
     [ae, ab] = earth_rad_eff(DN)
     This function computes the median value of the effective earth
     radius, and the effective Earth radius exceeded for beta0% of time
     as defined in ITU-R P.452-17.

     Input arguments:
     DN      -   the average radiorefractive index lapse-rate through the
                 lowest 1 km of the atmosphere (N-units/km)

     Output arguments:
     ae      -   the median effective Earth radius (km)
     ab      -   the effective Earth radius exceeded for beta0 % of time

     Example:
     [ae, ab] = earth_rad_eff(DN)

     Rev   Date        Author                          Description
     -------------------------------------------------------------------------------
     v0    18MAR22    Ivica Stevanovic, OFCOM         Initial implementation
    """
    
    k50 = 157/(157-DN)     #(5)

    ae = 6371*k50          #(6a)

    kbeta = 3

    ab = 6371*kbeta        #(6b)

    return ae, ab

def dl_se_ft_inner(epsr, sigma, d, hte, hre, adft, f):
    """
    dl_se_ft_inner The inner routine of the first-term spherical diffraction loss
    This function computes the first-term part of Spherical-Earth diffraction
    loss exceeded for p% time for antenna heights
    as defined in Sec. 4.2.2.1 of the ITU-R P.452-17, equations (30-37)
    
        Input parameters:
        epsr    -   Relative permittivity
        sigma   -   Conductivity (S/m)
        d       -   Great-circle path distance (km)
        hte     -   Effective height of interfering antenna (m)
        hre     -   Effective height of interfered-with antenna (m)
        adft    -   effective Earth radius (km)
        f       -   frequency (GHz)
    
        Output parameters:
        Ldft   -   The first-term spherical-Earth diffraction loss not exceeded for p% time
                    implementing equations (30-37), Ldft(1) is for horizontal
                    and Ldft(2) for the vertical polarization
    
        Example:
        Ldft = dl_se_ft_inner(epsr, sigma, d, hte, hre, adft, f)
    
        Rev   Date        Author                          Description
        -------------------------------------------------------------------------------
        v0    18MAR22     Ivica Stevanovic, OFCOM         Initial implementation
    """

    ## Body of the function

    # Normalized factor for surface admittance for horizontal (1) and vertical
    # (2) polarizations
    
    K = np.zeros(2)

    K[0] = 0.036* (adft*f)**(-1./3.) * ( (epsr-1)**2 + (18*sigma/f)**2. )**(-1./4.)   # Eq (30a)

    K[1] = K[0] * (epsr**2 + (18*sigma/f)**2)**(1./2.)       # Eq (30b)
    
    # Earth ground/polarization parameter

    beta_dft = (1 + 1.6*K**2 + 0.67* K**4) / ( 1 + 4.5* K**2 + 1.53* K**4)  # Eq (31)

    # Normalized distance

    X = 21.88* beta_dft * (f / adft**2)**(1./3.) * d          # Eq (32)

    # Normalized transmitter and receiver heights

    Yt = 0.9575* beta_dft * (f ** 2 / adft) ** (1/3) * hte       # Eq (33a)

    Yr = 0.9575* beta_dft * (f ** 2 / adft) ** (1/3) * hre       # Eq (33b)
 
    # Calculate the distance term given by:

    Fx = np.zeros(2)

    for ii in range(0,2):
        if X[ii] >= 1.6:
            Fx[ii] = 11 + 10*np.log10(X[ii]) - 17.6*X[ii]
        else:
            Fx[ii] = -20*np.log10(X[ii]) - 5.6488* (X[ii])**1.425     # Eq (34)
         

    Bt = beta_dft * Yt             # Eq (36b)

    Br = beta_dft * Yr              # Eq (36b)

    GYt = np.zeros(2)
    GYr = np.zeros(2)

    for ii in range(0,2):
        if Bt[ii]>2:
            GYt[ii] = 17.6*(Bt[ii] - 1.1) ** 0.5 - 5*np.log10(Bt[ii] -1.1)-8
        else:
            GYt[ii] = 20*np.log10(Bt[ii] + 0.1* Bt[ii] ** 3)
        
        if Br[ii]>2:
            GYr[ii] = 17.6*(Br[ii] - 1.1)**0.5 - 5*np.log10(Br[ii] -1.1)-8
        else:
            GYr[ii] = 20*np.log10(Br[ii] + 0.1* Br[ii] ** 3)
        
        
        if GYr[ii] < 2 + 20*np.log10(K[ii]):
            GYr[ii] = 2 + 20*np.log10(K[ii])
        
        
        if GYt[ii] < 2 + 20*np.log10(K[ii]):
            GYt[ii] = 2 + 20*np.log10(K[ii])
        
    Ldft = -Fx - GYt - GYr
   
    return Ldft

def dl_se_ft(d, hte, hre, adft, f, omega):
    """
    dl_se_ft First-term part of spherical-Earth diffraction according to ITU-R P.452-17
    This function computes the first-term part of Spherical-Earth diffraction
    loss exceeded for p% time for antenna heights
    as defined in Sec. 4.2.2.1 of the ITU-R P.452-17

    Input parameters:
    d       -   Great-circle path distance (km)
    hte     -   Effective height of interfering antenna (m)
    hre     -   Effective height of interfered-with antenna (m)
    adft    -   effective Earth radius (km)
    f       -   frequency (GHz)
    omega   -   fraction of the path over sea

    Output parameters:
    Ldft   -   The first-term spherical-Earth diffraction loss not exceeded for p% time
                Ldft(1) is for the horizontal polarization
                Ldft(2) is for the vertical polarization

    Example:
    Ldft = dl_se_ft(d, hte, hre, adft, f, omega)

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v0    18MAR22     Ivica Stevanovic, OFCOM         Initial implementation
    """


    ## Body of function

    # First-term part of the spherical-Earth diffraction loss over land

    epsr = 22
    sigma = 0.003

    Ldft_land = dl_se_ft_inner(epsr, sigma, d, hte, hre, adft, f)
    

    # First-term part of the spherical-Earth diffraction loss over sea

    epsr = 80
    sigma = 5

    Ldft_sea = dl_se_ft_inner(epsr, sigma, d, hte, hre, adft, f)


    # First-term spherical diffraction loss 

    Ldft = omega * Ldft_sea + (1-omega)*Ldft_land      # Eq (29)
 
    return Ldft
    end


def dl_se(d, hte, hre, ap, f, omega):
    """
    dl_se spherical-Earth diffraction loss exceeded for p% time according to ITU-R P.452-17
    This function computes the Spherical-Earth diffraction loss exceeded
    for p% time for antenna heights hte and hre (m)
    as defined in Sec. 4.2.2 of the ITU-R P.452-17

    Input parameters:
    d       -   Great-circle path distance (km)
    hte     -   Effective height of interfering antenna (m)
    hre     -   Effective height of interfered-with antenna (m)
    ap      -   the effective Earth radius in kilometers
    f       -   frequency expressed in GHz
    omega   -   the fraction of the path over sea

    Output parameters:
    Ldsph   -   The spherical-Earth diffraction loss not exceeded for p% time
                Ldsph(1) is for the horizontal polarization
                Ldsph(2) is for the vertical polarization

    Example:
    Ldsph = dl_se(d, hte, hre, ap, f, omega)

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v0    18MAR22     Ivica Stevanovic, OFCOM         Initial version
    """



    ## Body of function

    # Wavelength in meters

    lam = 0.2998/f

    # Calculate the marginal LoS distance for a smooth path

    dlos = np.sqrt(2*ap) * (np.sqrt(0.001*hte) + np.sqrt(0.001*hre))    # Eq (23)
    

    if d >= dlos:
        # calculate diffraction loss Ldft using the method in Sec. 4.2.2.1 for 
        # adft = ap and set Ldsph to Ldft
        
        Ldsph = dl_se_ft(d, hte, hre, ap, f, omega)
        
        return Ldsph
    else:
        # calculate the smallest clearance between the curved-Earth path and
        # the ray between the antennas, hse
        
        c = (hte - hre)/(hte + hre)        # Eq (25d)
        m = 250*d*d/(ap*(hte + hre))        # eq (25e)
        
        b = 2*np.sqrt((m+1.0)/(3.0*m)) * np.cos( np.pi/3 + 1.0/3.0* np.arccos( 3*c/2.0 * np.sqrt( 3.0*m/(m+1.0) ** 3 ) ) )   # Eq (25c)
        
        dse1 = d/2.0*(1.0+b)           # Eq (25a)
        dse2 = d - dse1            # Eq (25b) 
        
        hse = (hte - 500*dse1*dse1/ap)*dse2 + (hre - 500*dse2*dse2/ap)*dse1
        hse = hse/d                # Eq (24)
        
        # Calculate the required clearance for zero diffraction loss
        
        hreq = 17.456*np.sqrt(dse1 * dse2 * lam/d)     # Eq (26)
        
        
        if hse > hreq:
            Ldsph = np.zeros(2)
            return Ldsph
        else:
            
            # calculate the modified effective Earth radius aem, which gives
            # marginal LoS at distance d
            
            aem = 500*(d/( np.sqrt(hte) + np.sqrt(hre) )) ** 2     # Eq (27)
            
            # Use the method in Sec. 4.2.2.1 for adft ) aem to obtain Ldft
            
            Ldft = dl_se_ft(d, hte, hre, aem, f, omega)
            Ldft[Ldft < 0.0] = 0.0
            Ldsph = (1- hse/hreq)*Ldft     # Eq (28)
            
    
    return Ldsph
    
def dl_p( d, h, hts, hrs, hstd, hsrd, f, omega, p, b0, DN ):
    """
    dl_p Diffraction loss model not exceeded for p% of time according to P.452-17

    This function computes the diffraction loss not exceeded for p% of time
    as defined in ITU-R P.452-17 (Section 4.5.4)

        Input parameters:
        d       -   vector of distances di of the i-th profile point (km)
        h       -   vector of heights hi of the i-th profile point (meters
                    above mean sea level. Both vectors contain n+1 profile points
        hts     -   transmitter antenna height in meters above sea level (i=0)
        hrs     -   receiver antenna height in meters above sea level (i=n)
        hstd    -   Effective height of interfering antenna (m amsl) c.f. 5.1.6.3
        hsrd    -   Effective height of interfered-with antenna (m amsl) c.f. 5.1.6.3
        f       -   frequency expressed in GHz
        omega   -   the fraction of the path over sea
        p       -   percentage of time
        b0      -   the time percentage that the refractivity gradient (DELTA-N) exceeds 100 N-units/km in the first 100m of the lower atmosphere
        DN      -   the average radio-refractive index lapse-rate through the
                    lowest 1 km of the atmosphere. Note that DN is positive
                    quantity in this procedure

        Output parameters:
        Ldp    -   diffraction loss for the general path not exceeded for p % of the time 
                according to Section 4.2.4 of ITU-R P.452-17. 
                Ldp(1) is for the horizontal polarization 
                Ldp(2) is for the vertical polarization
        Ld50   -   diffraction loss for p = a50%

        Example:
        Ldp, Ld50 = dl_p( d, h, hts, hrs, hstd, hsrd, ap, f, omega, p, b0, DN )
        

        Rev   Date        Author                          Description
        -------------------------------------------------------------------------------
        v0    18MAR22     Ivica Stevanovic, OFCOM         Initial version

    """
    
    # Use the method in 4.2.3 to calculate diffractino loss Ld for effective 
    # Earth radius ap = ae as given by equation (6a). Set median diffractino
    # loss to Ldp50

    [ae, ab] = earth_rad_eff(DN)

    ap = ae

    Ld50 = dl_delta_bull( d, h, hts, hrs, hstd, hsrd, ap, f, omega )

    if p == 50:
        Ldp = Ld50
        return Ldp, Ld50
    

    if p < 50:
        
        # Use the method in 4.2.3 to calculate diffraction loss Ld for effective
        # Earth radius ap = abeta, as given in equation (6b). Set diffraction loss
        # not exceeded for beta0# time Ldb = Ld
        
        ap = ab
        
        Ldb = dl_delta_bull( d, h, hts, hrs, hstd, hsrd, ap, f, omega )

        # Compute the interpolation factor Fi
        
        if p > b0:
            
            Fi = inv_cum_norm(p/100) / inv_cum_norm(b0/100)   # eq (41a)
            
        else:
            
            Fi = 1
            
               
        # The diffraction loss Ldp not exceeded for p% of time is now given by
        
        Ldp = Ld50 + Fi*(Ldb - Ld50)   # eq (42)
        
    return Ldp, Ld50
    
def dl_delta_bull( d, h, hts, hrs, hstd, hsrd, ap, f, omega ):
    """
    dl_delta_bull Complete 'delta-Bullington' diffraction loss model P.452-17
    
    This function computes the complete 'delta-Bullington' diffraction loss
    as defined in ITU-R P.452-17 (Section 4.2.3)

        Input parameters:
        d       -   vector of distances di of the i-th profile point (km)
        h       -   vector of heights hi of the i-th profile point (meters
        above mean sea level. Both vectors contain n+1 profile points
        hts     -   transmitter antenna height in meters above sea level (i=0)
        hrs     -   receiver antenna height in meters above sea level (i=n)
        hstd    -   Effective height of interfering antenna (m amsl) c.f. 5.1.6.3
        hsrd    -   Effective height of interfered-with antenna (m amsl) c.f. 5.1.6.3
        ap      -   the effective Earth radius in kilometers
        f       -   frequency expressed in GHz
        omega   -   the fraction of the path over sea

        Output parameters:
        Ld     -   diffraction loss for the general patha according to
                Section 4.2.3 of ITU-R P.452-17. 
                Ld(1) is for the horizontal polarization 
                Ld(2) is for the vertical polarization

        Example:
        Ld = dl_delta_bull( d, h, hts, hrs, hstd, hsrd, ap, f, omega )
        

        Rev   Date        Author                          Description
        -------------------------------------------------------------------------------
        v0    01JAN16     Ivica Stevanovic, OFCOM         Initial version
    """

    # Use the method in 4.2.1 for the actual terrain profile and antenna
    # heights. Set the resulting Bullington diffraction loss for the actual
    # path to Lbulla

    Lbulla = dl_bull(d, h, hts, hrs, ap, f)

    # Use the method in 4.2.1 for a second time, with all profile heights hi
    # set to zero and modified antenna heights given by

    hts1 = hts - hstd   # eq (38a)
    hrs1 = hrs - hsrd   # eq (38b)
    h1 = np.zeros(h.shape)

    # where hstd and hsrd are given in 5.1.6.3 of Attachment 2. Set the
    # resulting Bullington diffraction loss for this smooth path to Lbulls

    Lbulls = dl_bull(d, h1, hts1, hrs1, ap, f)

    # Use the method in 4.2.2 to calculate the spherical-Earth diffraction loss
    # for the actual path length (dtot) with 

    hte = hts1             # eq (39a)
    hre = hrs1             # eq (39b)
    dtot = d[-1] - d[0]

    Ldsph = dl_se(dtot, hte, hre, ap, f, omega)

    # Diffraction loss for the general paht is now given by
    
    Ld = np.zeros(2)

    Ld[0] = Lbulla + max(Ldsph[0] - Lbulls, 0)  # eq (40)
    Ld[1] = Lbulla + max(Ldsph[1] - Lbulls, 0)  # eq (40)

    return Ld
    

def dl_bull(d, h, hts, hrs, ap, f):
    """
    dl_bull Bullington part of the diffraction loss according to P.452-17
    This function computes the Bullington part of the diffraction loss
    as defined in ITU-R P.452-17 in 4.2.1

        Input parameters:
        d       -   vector of distances di of the i-th profile point (km)
        h       -   vector of heights hi of the i-th profile point (meters
        above mean sea level. Both vectors contain n+1 profile points
        hts     -   transmitter antenna height in meters above sea level (i=0)
        hrs     -   receiver antenna height in meters above sea level (i=n)
        ap      -   the effective earth radius in kilometers
        f       -   frequency expressed in GHz

        Output parameters:
        Lbull   -   Bullington diffraction loss for a given path

        Example:
        Lbull = dl_bull(d, h, hts, hrs, ap, f)

        Rev   Date        Author                          Description
        -------------------------------------------------------------------------------
        v0    23DEC15     Ivica Stevanovic, OFCOM         First implementation in matlab
    """

    # Effective Earth curvature Ce (km^-1)

    Ce = 1/ap

    # Wavelength in meters

    lam = 0.2998/f

    # Complete path length

    dtot = d[-1]-d[0]

    # Find the intermediate profile point with the highest slope of the line
    # from the transmitter to the point

    di = d[1:-1]
    hi = h[1:-1]

    Stim = max((hi + 500*Ce*di * (dtot - di) - hts) / di )           # Eq (14)

    # Calculate the slope of the line from transmitter to receiver assuming a
    # LoS path

    Str = (hrs - hts)/dtot                                         # Eq (15)

    if Stim < Str: # Case 1, Path is LoS
        
        # Find the intermediate profile point with the highest diffraction
        # parameter nu:
        
        numax = max (\
                    ( hi + 500*Ce*di * (dtot - di) - ( hts*(dtot - di) + hrs*di)/dtot ) * \
                    np.sqrt(0.002*dtot / (lam*di*(dtot-di))) \
                    )
                
        Luc = 0
        if numax > -0.78:
            Luc = 6.9 + 20*np.log10(np.sqrt((numax-0.1) ** 2+1) + numax - 0.1)   # Eq (13), (17)
        
    else:
        
        # Path is transhorizon
        
        # Find the intermediate profile point with the highest slope of the
        # line from the receiver to the point
        
        Srim = max((hi + 500*Ce*di * (dtot-di)-hrs) / (dtot-di))     # Eq (18)
        
        # Calculate the distance of the Bullington point from the transmitter:
        
        dbp = (hrs - hts + Srim*dtot)/(Stim + Srim)                # Eq (19)
        
        # Calculate the diffraction parameter, nub, for the Bullington point
        
        nub =  ( hts + Stim*dbp - ( hts*(dtot - dbp) + hrs*dbp)/dtot ) * \
                    np.sqrt(0.002*dtot/(lam*dbp*(dtot-dbp)))    # Eq (20)
        
        # The knife-edge loss for the Bullington point is given by
                
        Luc = 0
        if nub > -0.78:
            Luc = 6.9 + 20*np.log10(np.sqrt((nub-0.1) ** 2+1) + nub - 0.1)   # Eq (13), (21)
        
    
    # For Luc calculated using either (17) or (21), Bullington diffraction loss
    # for the path is given by

    Lbull = Luc + (1 - np.exp(-Luc/6.0))*(10+0.02*dtot)         # Eq (22)
    return Lbull
    
def closs_corr(f, d, h, zone, htg, hrg, ha_t, ha_r, dk_t, dk_r):
    """
    closs_corr: clutter loss correction according to P.452-17

    This function computes the height-gain correction as defined in ITU-R P.452-17 (Section 4.5.4)

    Input parameters:
    f       -   Frequency (GHz)
    d       -   vector of distances di of the i-th profile point (km)
    h       -   vector of heights hi of the i-th profile point (meters
                above mean sea level. Both vectors contain n+1 profile points
    zone    -   Zone type: Coastal land (1), Inland (2) or Sea (3)
    htg     -   Tx Antenna center heigth above ground level (m)
    hrg     -   Rx Antenna center heigth above ground level (m)
    ha_t    -   Nominal clutter height at the transmitting end (m, agl)
    ha_r    -   Nominal clutter height at the receiving end (m, agl)
    dk_t    -   distance from nominal clutter point to the Tx antenna (km)
    dk_r    -   distance from nominal clutter point to the Rx antenna (km)

    Output parameters:
    dc      -   vector of distances in the height-gain model
    hc      -   vector of heights in the height-gain model
    zonec   -   Zone type: Coastal land (1), Inland (2) or Sea (3)in the height-gain model
    htgc    -   Tx Antenna center heigth above ground level (m) in the height-gain model
    hrgc    -   Rx Antenna center heigth above ground level (m) in the height-gain model
    Aht     -   Additional losses to account for clutter shielding the
    Ahr         transmitter and receiver. These should be set to zero if there is no
                such shielding

    Example:
    dc,hc,zonec,htgc,hrgc, Aht, Ahr = closs_corr(d, h, zone, htg, hrg, ha_t, ha_r, dk_t, dk_r)


    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v0    18MAR22     Ivica Stevanovic, OFCOM         Initial version
    """


    index1 = 0
    index2 = len(d)-1

    htgc = htg
    hrgc = hrg

    Aht = 0
    Ahr = 0

    ha = ha_t
    dk = dk_t

    if ha > htg:
        
        Ffc = 0.25+0.375*(1+np.tanh(7.5*(f-0.5)))  # (57a)
        
        Aht = 10.25*Ffc*np.exp(-dk)*( 1 - np.tanh(6*(htg/ha-0.625)) )-0.33 # (57)
        
        flagAht = 1
        
       
        (kk,) = np.where(d>=dk)
        
        if (~isempty(kk)):
            index1 = kk[0]
        else:
            index1 = len(d)
        
        htgc = ha_t

    
    ha = ha_r
    dk = dk_r

    if ha > hrg:
        
        Ffc = 0.25+0.375*(1+np.tanh(7.5*(f-0.5)))  # (57a)
        
        Ahr = 10.25*Ffc*np.exp(-dk)*( 1 - np.tanh(6*(hrg/ha-0.625)) )-0.33  # (57)
        
        flagAhr = 1
        
        (kk,) = np.where(d <= d[-1]-dk)
        if(~isempty(kk)):
            index2 = kk[-1]
        else:
            index2 = 0
        
        
        hrgc = ha_r
    

    # Modify the path

    if (index2-index1 < 3): # at least two points between the clutter at Tx and Rx sides
        raise ValueError('Py452: closs_corr: the sum of clutter nominal distances is larger than the path length.')
    
 
    dc = d[index1:index2+1]-d[index1]
    hc = h[index1:index2+1]
    zonec = zone[index1:index2+1]

    return dc, hc, zonec,htgc, hrgc, Aht, Ahr
    
def beta0(phi, dtm, dlm):
    """
    This function computes the time percentage for which refractive index
    lapse-rates exceeding 100 N-units/km can be expected in the first 100
    m of the lower atmosphere
    as defined in ITU-R P.452-17.

    Input arguments:
    phi     -   path centre latitude (deg)
    dtm     -   the longest continuous land (inland + coastal) section of the great-circle path (km)
    dlm     -   the longest continuous inland section of the great-circle path (km)

    Output arguments:
    b0      -   the time percentage that the refractivity gradient (DELTA-N) exceeds 100 N-units/km in the first 100m of the lower atmosphere

    Example:
    b0 = beta0(phi, dtm, dlm)

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v0    18MAR22     Ivica Stevanovic, OFCOM         Initial implementation
    """

    tau = 1- np.exp(-(4.12*1e-4*dlm**2.41))       # (3a)

    mu1 = ( 10**(-dtm/(16-6.6*tau)) +  10**(-5*(0.496 + 0.354*tau)) ) ** 0.2 # (3)

    if mu1 > 1:
        mu1 = 1
    
    
    if np.abs(phi) <= 70:
        mu4 = 10**( (-0.935 + 0.0176*np.abs(phi))*np.log10(mu1) )   # (4)
        b0 =  10**( -0.015*abs(phi) + 1.67 )*mu1*mu4           # (2)   
    else:
        mu4 = 10**(0.3*np.log10(mu1))                            # (4)
        b0 = 4.17*mu1*mu4                                    # (2)
    

    return b0
    
         
def isempty(x):
    if np.size(x) == 0:
        return True
    else:
        return False