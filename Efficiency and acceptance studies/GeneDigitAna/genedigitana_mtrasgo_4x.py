# -*- coding: utf-8 -*-
"""
.../genedigitana_mtrasgo_4x.py
!!!!!
Version con tsum y tdif, 4 strips X asimetricos
Juan A. Garzon. LabCAF-USC
Junio. 2023
6 parameter track fitting (X0, XP, Y0, YP, T0, S0)
All strips parallel along the x-axis
The program is divided in four sections:
    0. Detector layout
    1. Generation of tracks
    2. Digitization of the tracks in the detector
    3. Analyis and reconstruction of the generated tracks
Resolutions given by Alberto Blanco:    
    TDC: 16ps
    FEE + TDC: 35ps
    sigX0: 60ps  (X0 = lenx/2 + vs · dtE/2 => sigX0 = vs·sig(dtE)/2 => sigtE ~85ps)
    sigT0: 300ps (tsum = 2·T0 + ss·lenx => sigT0 = sigTsum / sqrt2)
We assume: sigT = sigTbaseline + sig(FEE + TDC)
"""
import numpy as np
import scipy.linalg as linalg
from math import sqrt,pi,floor
import matplotlib.pyplot as plt
#import time

# A few important functions
def fmgx(nvar, npar, vs, ss, zi): # G matrix for t measurements in X-axis
    mg = np.zeros([nvar, npar])
    XP = vs[1]; YP = vs[3]; S0 = vs[5]
    kz = sqrt(1 + XP*XP + YP*YP)
    kzi = 1 / kz
    #
    mg[0,2] = 1
    mg[0,3] = zi
    mg[1,1] = kzi * S0 * XP * zi
    mg[1,3] = kzi * S0 * YP * zi
    mg[1,4] = 1
    mg[1,5] = kz * zi
    mg[2,0] = ss
    mg[2,1] = ss * zi
    return mg
def fmwx(nvar, sy, sts, std): # Weigth matrix 
    mw = np.zeros([nvar, nvar])
    mw[0,0] = 1/(sy*sy)
    mw[1,1] = 1/(sts*sts)
    mw[2,2] = 1/(std*std)
    return mw
def fvmx(nvar, vs, lenx, ss, zi): # Fitting model array with X-strips
    vm = np.zeros(nvar)
    X0 = vs[0]; XP = vs[1]; Y0 = vs[2]; YP = vs[3]; T0 = vs[4]; S0 = vs[5]
    kz = np.sqrt(1 + XP*XP + YP*YP)
    #
    xi = X0 + XP * zi
    yi = Y0 + YP * zi
    ti = T0 + kz * S0 * zi
    th = 0.5 * lenx * ss   # tau half
    lxmn = -lenx/2
    #
    vm[0] = yi
    vm[1] = th + ti
    vm[2] = ss * (xi-lxmn) - th
    return vm
def fmkx(nvar, npar, vs, sy, sts, std, ss, zi): # K matrix
    mk  = np.zeros([npar,npar])
    mg  = fmgx(nvar, npar, vs, ss, zi)
    mgt = mg.transpose()
    mw  = fmwx(nvar, sy, sts, std)
    mk  = mgt @ mw @ mg
    return mk
def fvax(nvar, npar, vs, vdat, sy, sts, std, lenx, ss, zi): # va vector
    va= np.zeros(npar)
    mw = fmwx(nvar, sy, sts, std)
    vm = fvmx(nvar, vs, lenx, ss, zi)
    mg = fmgx(nvar, npar, vs, ss, zi)
    vdmg = vdat - (vm - mg @ vs)
    va = mg.transpose() @ mw @ vdmg
    return va
def fmahd(npar, vin1, vin2, merr): # Mahalanobis distance
    #vdif = np.ones(npar)
    #vdsq = np.ones(npar)
    #vsig = np.ones(npar)
    #
    vdif  = np.subtract(vin1,vin2)
    vdsq  = np.power(vdif,2)
    verr  = np.diag(merr,0)
    vsig  = np.divide(vdsq,verr)
    dist  = np.sqrt(np.sum(vsig))
    return dist
# ============================================================================
# ============================================================================
# ============================================================================
#
ntrk  = 1
cocut = 1   # convergence cut
efs   = 1
# Constants	
vc = 300 #mm/ns
sc = 1/vc
vp  = 0.6 * vc  # velocity of the signal in the strip
ss  = 1/vp
sq2 = sqrt(2)
sq2i  = 1./sq2
sq12i = 1./sqrt(12)
d2r = pi/180 # from degrees to radians
r2d = 1/d2r
stht  = 10 # sigma of theta angle
thmx  = 60 # theta max
thmxr = thmx * d2r # theta max in radian
cthmx = np.cos(thmxr)
npar = 6  # No. of parameters: [X0,XP,Y0,YP,T0,S0]
nvar = 3  # tdif, tsum, ycoor
#
# test particle
xini  = 60
yini  = 60
slopx = 0.0 #sin(thang) * cos(phang)
slopy = 0.0 #sin(thang) * sin(phang)
t0    = 1000
# Detector layout
# We assume a detector with Nplan square shaped planes with strips parallel to the X-axe
lenx  = 300;
leny  = 300;
nst   = 4;         # number of strips/plane
#wst   = leny/nst;  # mm, width of a strip
#wsth  = wst/2
lxmn  = -lenx/2;   # Origin of coordinates
lymn  = -lenx/2;
lxmx  = lenx/2;
lymx  = lenx/2;
#tau   = lenx * ss;
#
#sy    = wst*sq12i;
stb   = 0.3; # sigt contribution by the base line fluctuation
ste   = 0.085; # sigt contribution by the electronics fluctuation
tdc   = 0.016
#tdcr  = 1 # tdc resolution
sts   = sqrt(stb*stb + sq2i*ste*ste);
std   = ste * sq2i
#
# Important arrays
nplan = 4;
nstrp = 4   # nb. of strips
vz    = np.zeros(nplan) # z coordinates of the planes
vx    = np.zeros(nplan) # x coordinate of the particle at each plane
vy    = np.zeros(nplan) # y coordinate of the particle at each plane
vtd   = np.zeros(nplan) # Digitized time coordinate
vyd   = np.zeros(nplan) # Digitized y coordinate
mstrp = np.zeros([nplan, nstrp]) 
mwdts = np.zeros([nplan, nstrp])
#
vz    = [0,100,200,400]
zor   = vz[0];
#
mstrp = [[5, 69, 133, 197],
        [5, 104, 168, 232],
        [5, 69, 133, 197],
        [5, 104, 168, 232]]
mwdts = [[63, 63, 63, 98],
         [98, 63, 63, 63],
         [63, 63, 63, 98],
         [98, 63, 63, 63]]
#
# Track generation ==================================================== GENE    
vps  = np.zeros(ntrk) # Vector of particle velocity
vth  = np.zeros(ntrk) # Vector of theta angles
vct  = np.zeros(ntrk) # Vector of theta angles
vph  = np.zeros(ntrk) # Vector of phi angles
vtin = np.zeros(ntrk) # Vector of initial time
vxin = np.zeros(ntrk)
vyin = np.zeros(ntrk)
vslx = np.zeros(ntrk)
vsly = np.zeros(ntrk)
#
vx0i = np.array(0)
vxpi = np.array(0)
vy0i = np.array(0)
vypi = np.array(0)
vt0i = np.array(0)
vs0i = np.array(0)
vv0i = np.array(0)
vtha = np.array(0)
vcta = np.array(0)
#
vx0f = np.array(0)
vxpf = np.array(0)
vy0f = np.array(0)
vypf = np.array(0)
vt0f = np.array(0)
vs0f = np.array(0)
vv0f = np.array(0)
#
vdx0 = np.array(0)
vdxp = np.array(0)
vdy0 = np.array(0)
vdyp = np.array(0)
vdt0 = np.array(0)
vds0 = np.array(0)
vdv0 = np.array(0)
#
vsx0 = np.array(0)
vsxp = np.array(0)
vsy0 = np.array(0)
vsyp = np.array(0)
vst0 = np.array(0)
vss0 = np.array(0)
vsv0 = np.array(0)
#
mtag = np.zeros([ntrk,nplan])
mhit = np.zeros([ntrk,nplan,nvar]) # Matrix of tracks data
mhid = np.zeros([ntrk,nplan,nvar]) # Matrix of digitized tracks
mdat = np.zeros([ntrk,nplan,nvar]) # Matrix of digitized tracks
vdat = np.zeros(nvar)

beta  = 0.999
for it in range(ntrk):    # Loop on track number
    vel  = vc * beta
    slw  = 1/vel  # slowness    
    vps[it] = slw
    #  Uniform random incident position
    '''
    xini = np.random.uniform(lxmn,lxmx) #np.random.normal(0,1)*5*lenx/2-xor
    yini = np.random.uniform(lymn,lymx) #np.random.normal(0,1)*5*leny/2-yor
    '''
    vxin[it] = xini #np.random.uniform(lxmn,lxmx) #np.random.normal(0,1)*5*lenx/2-xor
    vyin[it] = yini 
    vtin[it] = t0
    '''
    #  cosTheta random distribution between 0 and thmax
    cthan  = np.random.uniform(cthmx,1)
    thang  = np.arccos(cthan)*r2d
    thangr = thang * d2r
    '''
    #  cosTheta^2 distribution between 0 and thmax
    #cthmx=0
    cthan  = np.random.uniform(0,1)**0.25
    cthan  = 1    # ---------------------------------------
    thangr = np.arccos(cthan)
    thang  = thangr * r2d
    #
    phang   = np.random.uniform(0,1) * 360 
    phang  = 0    # ---------------------------------------
    phangr  = phang * d2r
    vth[it] = thang
    vct[it] = cthan
    vph[it] = phang   
    slopx   = np.sin(thangr) * np.cos(phangr)
    slopy   = np.sin(thangr) * np.sin(phangr)
    vslx[it]= slopx
    vsly[it]= slopy
    #print('---', thang, phang)
    #print('---',xini, yini, slopx, slopy)
#    r0[it] = np.array([x0[it],y0[it],z0[it]]).reshape(3)
#    
# Loop on planes for digitizing the trackz ========================== DIGIT
for it in range(ntrk):
    #vtag = [0, 0, 0, 0]
    x0 = vxin[it]
    xp = vslx[it]
    y0 = vyin[it]
    yp = vsly[it]
    t0 = vtin[it]
    s0 = vps[it]
    #
    kz = sqrt(1 + xp**2 + yp**2)
    for ip in range(nplan):
        zi = vz[ip]
        xi = x0 + xp * zi 
        yi = y0 + yp * zi 
        ti = t0 + kz * s0 * zi
        tf = ti + (xi-lxmn) * ss
        tb = ti + (lxmx-xi)* ss
        ts = 0.5 * (tf + tb)
        td = 0.5 * (tf - tb)
        mhit[it,ip]=[yi, ts,td]
        # Digitization
        #xid = floor((xi-lxmn)/wst) * wst + wst/2
        for ist in range (nstrp):
            sbl = np.random.normal(0,1) * stb # t0 + common base-line fluctuation
            sef = np.random.normal(0,1) * ste # t0 + fee time fluctuations
            seb = np.random.normal(0,1) * ste
            tfs  = tf + sbl + sef
            tbs  = tb + sbl + seb
            tfd = floor(tfs/tdc)*tdc + tdc/2
            tbd = floor(tbs/tdc)*tdc + tdc/2
            tsd = 0.5 * (tfd + tbd)   # 
            tdd = 0.5 * (tfd - tbd)
            #
            ystp = mstrp[ip][ist] + lymn
            ytrk = yi
            #print(ytrk, ystp)
            if (ystp > ytrk): continue
            wstr = mwdts[ip][ist]
            dyst = ytrk - ystp
            #print(ip, ist, dyst, wstr)
            efr = np.random.binomial(1,efs)
            if(dyst > wstr or efr == 0): continue
            yhit = ystp + wstr/2
            ihit = ist
            mtag[it,ip] = 1   # particle is going through an active region
            #print(ip, ist, ytrk, ystp, yhit)
            mhid[it,ip]=[yhit, tsd, tdd]
            mdat[it,ip] = [ihit, tsd, tdd]
            #print(it, ip, ist, ts, tsd, td, tdd)

# Analysis step ======================================================
#    Saeta is a set of all the fitting parameters vs=[X0,XP,Y0,YP,T0,S0] 
#    Starting saeta for the iterative fit  
X0i = 0.0
XPi = 0.0
Y0i = 0.0
YPi = 0.0
T0i = 0.0
S0i = sc
kz  = sqrt(1 + XPi**2 + YPi**2)
dist = 10
nft = 0
#
for it in range(ntrk):
    nfip = sum(mtag[it,:])   # nb. of fired planes
    #print('***', nfip, mtag[it,:])
    if nfip > 1:
        x0  = X0i
        xp  = XPi
        y0  = Y0i
        yp  = YPi
        t0  = T0i
        s0  = S0i
        vs = [x0,xp,y0,yp,t0,s0]
        vsi  = np.asarray([vxin[it],vslx[it],vyin[it],vsly[it],vtin[it],vps[it]])
        #print('-vsi:',vsi)
        mk  = np.zeros([npar, npar])
        mki = np.zeros([npar, npar])
        va  = np.zeros(npar)
        vsf = np.zeros(npar)
        istp = 0
        #
        while dist>cocut:
            for ip in range(nplan):
                if(mtag[it,ip]== 0): continue
                tha  = vth[it]
                ctha = vct[it]
                zi   = vz[ip]
                #
                yst  = mdat[it][ip][0]
                tsum = mdat[it,ip,1]
                tdif = mdat[it,ip,2]
                #
                #istr = np.int(mdat[it][ip][0])
                istr = np.int(yst)
                wstr = mwdts[ip][istr]
                sy  = wstr * sq12i
                mk = mk + fmkx(nvar, npar, vs, sy, sts, std, ss, zi)
                vdat[0] = mstrp[ip][istr] + mwdts[ip][istr]/2 + lymn
                vdat[1] = tsum
                vdat[2] = tdif
                #print(zi, istp, vdat)
                va = va + fvax(nvar, npar, vs, vdat, sy, sts, std, lenx, ss, zi)
            #
            istp = istp + 1
            merr = linalg.inv(mk)     # Error matrix
            vsf  = merr @ va          # sEa equation
            dist = fmahd(npar, vs, vsf, merr)
            # print("***** ", istp, dist)
            vs = vsf
        dist = 10#
        nft = nft + 1  # nb. of fitted tracks
        # Uncertainties
        vx0i = np.append(vx0i, vsi[0])
        vxpi = np.append(vxpi, vsi[1])
        vy0i = np.append(vy0i, vsi[2])
        vypi = np.append(vypi, vsi[3])
        vt0i = np.append(vt0i, vsi[4])
        vs0i = np.append(vs0i, vsi[5])
        vv0i = np.append(vv0i, 1/vsi[5])
        vtha = np.append(vtha, tha)
        vcta = np.append(vcta, ctha)
        #
        #vx0f[nft] = vsf[0]
        vx0f = np.append(vx0f, vsf[0])
        vxpf = np.append(vxpf, vsf[1])
        vy0f = np.append(vy0f, vsf[2])
        vypf = np.append(vypf, vsf[3])
        vt0f = np.append(vt0f, vsf[4])
        vs0f = np.append(vs0f, vsf[5])
        vv0f = np.append(vv0f, 1/vsf[5])
        #
        vsx0 = np.append(vsx0, sqrt(merr[0][0]))
        vsxp = np.append(vsxp, sqrt(merr[1][1]))
        vsy0 = np.append(vsy0, sqrt(merr[2][2]))
        vsyp = np.append(vsyp, sqrt(merr[3][3]))
        vst0 = np.append(vst0, sqrt(merr[4][4]))
        vss0 = np.append(vss0, sqrt(merr[5][5]))
        #
        '''
        fs0 = vsf[5]   # fitted slowness
        fv0 = 1/fs0    # fitted velocity
        sfv = vss0[nft]/fs0 * fv0 # velocity uncertainty
        vsv0[nft] = sfv
        '''
        #
        #print('-----', vsi)
        #print('-----', vsf)
        #print('*****', vdx0[nft-1], vdy0[nft-1])
        #print()
#        
vx0i = np.delete(vx0i,(0))
vxpi = np.delete(vxpi,(0))
vy0i = np.delete(vy0i,(0))
vypi = np.delete(vypi,(0))
vt0i = np.delete(vt0i,(0))
vs0i = np.delete(vs0i,(0))
vv0i = np.delete(vv0i,(0))
vtha = np.delete(vtha,(0))
vcta = np.delete(vcta,(0))
#
vx0f = np.delete(vx0f,(0))
vxpf = np.delete(vxpf,(0))
vy0f = np.delete(vy0f,(0))
vypf = np.delete(vypf,(0))
vt0f = np.delete(vt0f,(0))
vs0f = np.delete(vs0f,(0))
vv0f = np.delete(vv0f,(0))
#
vsx0 = np.delete(vsx0,(0))
vsxp = np.delete(vsxp,(0))
vsy0 = np.delete(vsy0,(0))
vsyp = np.delete(vsyp,(0))
vst0 = np.delete(vst0,(0))
vss0 = np.delete(vss0,(0))
#
vdx0 = vx0i - vx0f
vdxp = vxpi - vxpf
vdy0 = vy0i - vy0f
vdyp = vypi - vypf
vdt0 = vt0i - vt0f
vds0 = vs0i - vs0f
vdv0 = vv0i - vv0f
'''
#
#   Summary plots
# 
plt.close('all')
#
ntr   = nft
nbins = 50
plt.figure(1)
vh = vdx0
x0mx = np.max(np.abs(vh))
x0mn = np.mean(vh)
x0md = np.median(vh)
x0sd = np.std(vh)
stat=f"{ntr} events \n\
Mean = {x0mn:.4g} \n\
Std. dev. = {x0sd:.4g}"
plt.hist(vh, bins=nbins, alpha=0.7, label = stat)
#plt.hist(vh,bins=50)
#plt.grid(True)
plt.title(' dX0 residuals. 100k events. Eff=0.95')
plt.yscale('log')
plt.xlabel('dX0',size=14)
plt.ylabel('# Entries',size=14)
plt.legend()
plt.show()
#
plt.figure(2)
vh = vdxp
xpmx = np.max(np.abs(vh))
xpmn  = np.mean(vh)
xpmd  = np.median(vh)
xpsd  = np.std(vh)
stat=f"{ntr} events \n\
Mean = {xpmn:.4g} \n\
Std. dev. = {xpsd:.4g}"
plt.hist(vh, bins=nbins, alpha=0.7, label = stat)
#plt.grid(True)
plt.title(' dXP residuals. 100k events. Eff=0.95')
plt.yscale('log')
plt.xlabel('XP',size=14)
plt.ylabel('# Entries',size=14)
plt.legend()
plt.show()
#
plt.figure(3)
vh = vdy0
y0mx = np.max(np.abs(vh))
y0mn  = np.mean(vh)
y0md  = np.median(vh)
y0sd  = np.std(vh)
stat=f"{ntr} events \n\
Mean = {y0mn:.4g} \n\
Std. dev. = {y0sd:.4g}"
plt.hist(vh, bins=nbins, alpha=0.7, label = stat)
#plt.grid(True)
plt.title(' Y0 residuals. 100k events. Eff=0.95')
plt.yscale('log')
plt.xlabel('dY0',size=14)
plt.ylabel('# Entries',size=14)
plt.legend()
plt.show()
#
plt.figure(4)
vh = vdyp
ypmx = np.max(np.abs(vh))
ypmn  = np.mean(vh)
ypmd  = np.median(vh)
ypsd  = np.std(vh)
stat=f"{ntr} events \n\
Mean = {ypmn:.4g} \n\
Std. dev. = {ypsd:.4g}"
plt.hist(vh, bins=nbins, alpha=0.7, label = stat)
#plt.grid(True)
plt.title(' YP residuals. 100k events. Eff=0.95')
plt.yscale('log')
plt.xlabel('dYP',size=14)
plt.ylabel('# Entries',size=14)
plt.legend()
plt.show()
#
plt.figure(5)
vh = vdt0
t0mx = np.max(np.abs(vh))
t0mn  = np.mean(vh)
t0md  = np.median(vh)
t0sd  = np.std(vh)
stat=f"{ntr} events \n\
Mean = {t0mn:.4g} \n\
Std. dev. = {t0sd:.4g}"
plt.hist(vh, bins=nbins, alpha=0.7, label = stat)
#plt.grid(True)
plt.title(' T0 residuals. 100k events. Eff=0.95')
plt.yscale('log')
plt.xlabel('dT0',size=14)
plt.ylabel('# Entries',size=14)
plt.legend()
plt.show()
#
plt.figure(6)
vh = vds0
dsmx = np.max(np.abs(vh))
dsmn  = np.mean(vh)
dsmd  = np.median(vh)
dssd  = np.std(vh)
stat=f"{ntr} events \n\
Mean = {dsmn:.4g} \n\
Std. dev. = {dssd:.4g}"
plt.hist(vh, bins=nbins, alpha=0.7, label = stat)
#plt.grid(True)
plt.title(' S0 residuals. 100k events. Eff=0.95')
plt.yscale('log')
plt.xlabel('dS0',size=14)
plt.ylabel('# Entries',size=14)
plt.legend()
plt.show()
#
plt.figure(7)
vh = vs0f
s0mx = np.max(np.abs(vh))
s0mn  = np.mean(vh)
s0md  = np.median(vh)
s0sd  = np.std(vh)
stat=f"{ntr} events \n\
Mean = {s0mn:.4g} \n\
Std. dev. = {s0sd:.4g}"
plt.hist(vh, bins=nbins, color='red',alpha=0.7, label = stat)
#plt.grid(True)
plt.title(' S0 distribution. 100k events. Eff=0.95')
plt.yscale('log')
plt.xlabel('S0_fit',size=14)
plt.ylabel('# Entries',size=14)
plt.legend()
plt.show()
#
plt.figure(11)
plt.clf()
plt.hexbin(vtha,vdxp,gridsize=200)
plt.xlim(0,thmx)
plt.ylim(-0.1,0.1)
plt.xlabel ('Zenith Angle/º')
plt.ylabel ('dXP')
plt.title('XP residual vs. incident zenith angle')
#
plt.figure(12)
plt.clf()
plt.hexbin(vtha,vx0i,gridsize=200)
plt.xlim(0,thmx)
plt.ylim(-150,150)
plt.xlabel ('Zenith Angle/º')
plt.ylabel ('X0')
plt.title('X0 vs. incident zenith angle')
#
plt.figure(13)
plt.clf()
plt.hexbin(vcta,vx0i, gridsize=200)
plt.xlim(0.8,1)
plt.ylim(-150,150)
plt.xlabel ('Cos Zenith Angle/º')
plt.ylabel ('X0')
plt.title('X0 vs. Cos incident zenith angle')
'''
'''
#
plt.figure(78
vh = vdv0
v0mx = np.max(np.abs(vh))
v0mn  = np.mean(vh)
v0md  = np.median(vh)
v0sd  = np.std(vh)
stat=f"{ntr} events \n\
Mean = {x0mn:.4g} \n\
Std. dev. = {x0sd:.4g}"
plt.hist(vh, bins=nbins, alpha=0.7, label = stat)
#plt.grid(True)
plt.title(' dV0 distribution')
plt.yscale('log')
plt.xlabel('dV0',size=14)
plt.ylabel('# Entries',size=14)
plt.legend()
plt.show()
#
'''

