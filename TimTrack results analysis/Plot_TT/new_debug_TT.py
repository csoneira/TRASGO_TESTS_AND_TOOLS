#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:28:00 2024

@author: cayesoneira
"""

def tt_nico(filename):
    # filename = "timtrack_data_ypos_cal_pos_cal_time.bin"

    Limit = False
    limit_number = 10000
    import numpy as np
    import scipy.linalg as linalg
    import scipy.stats as stats
    from math import sqrt
    #import time
    #%% A few important functions
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
    def fmwx(nvar, vsig): # Weigth matrix 
        sy = vsig[0];  sts=vsig[1];  std =vsig[2]
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
    def fmkx(nvar, npar, vs, vsig, ss, zi): # K matrix
        mk  = np.zeros([npar,npar])
        mg  = fmgx(nvar, npar, vs, ss, zi)
        mgt = mg.transpose()
        mw  = fmwx(nvar, vsig)
        mk  = mgt @ mw @ mg
        return mk
    def fvax(nvar, npar, vs, vdat, vsig, lenx, ss, zi): # va vector
        va = np.zeros(npar)
        mw = fmwx(nvar, vsig)
        vm = fvmx(nvar, vs, lenx, ss, zi)
        mg = fmgx(nvar, npar, vs, ss, zi)
        vg = vm - mg @ vs
        vdmg = vdat - vg
        va = mg.transpose() @ mw @ vdmg
        return va
    def fs2(nvar, npar, vs, vdat, vsig, lenx, ss, zi):
        va = np.zeros(npar)
        mk = fmkx(nvar, npar, vs, vsig, ss, zi)
        va = fvax(nvar, npar, vs, vdat, vsig, lenx, ss, zi)
        vm = fvmx(nvar, vs, lenx, ss, zi)
        mg = fmgx(nvar, npar, vs, ss, zi)
        mw = fmwx(nvar, vsig)
        vg = vm - mg @ vs
        vdmg = vdat - vg
        #
        mg = fmgx(nvar, npar, vs, ss, zi)
        sk = vs.transpose() @ mk @ vs
        sa = vs.transpose() @ va
        s0 = vdmg.transpose() @ mw @ vdmg
        s2 = sk - 2*sa + s0
        return s2
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
    def fres(vs, vdat, vsig, lenx, ss, zi):  # Residuals array
        sy = vsig[0]; sts = vsig[1]; std = vsig[2]
        X0 = vs[0]; XP = vs[1]; Y0 = vs[2]; YP = vs[3]; T0 = vs[4]; S0 = vs[5]
        kz = sqrt(1 + XP*XP + YP*YP)
        #  Fitted values
        xf  = X0
        # xf  = X0 + XP * zi
        yf  = Y0 + YP * zi
        # tff = T0 + S0 * kz * zi + (lenx/2 + xf) * ss
        # tbf = T0 + S0 * kz * zi + (lenx/2 - xf) * ss
        # tsf = 0.5 * (tff + tbf)
        # tdf = 0.5 * (tff - tbf)
        
        # tsf = T0 + S0 * kz * zi
        # tsf = T0 + S0 * kz * zi + lenx / 2 * ss
        tsf = T0 + lenx / 2 * ss
        # tsf = S0 * kz * zi + lenx / 2 * ss
        # tdf = 2 * xf * ss
        tdf = xf * ss
        
        #  Data
        yd  = vdat[0]
        tsd = vdat[1]
        tdd = vdat[2]
        
        # print(tsd)
        
        # print("--------------------------`
        
        # Residuals
        yr   = (yf - yd)/sy
        tsr  = (tsf  - tsd)/sts
        tdr  = (tdf  - tdd)/std
        
        # yr   = (yf - yd)
        # tsr  = (tsf  - tsd)
        # tdr  = (tdf  - tdd)
        # Residuals array
        vres = [yr, tsr, tdr]
        return vres
    #%%#   A few constants
    vc  = 300 #mm/ns
    sc  = 1/vc
    vp  = 2/3 * vc  # velocity of the signal in the strip
    ss  = 1/vp
    #
    cocut = 1  # convergence cut
    d0    = 10 # initial value of the convergence parameter 
    #%% Detector layout
    nplan = 4
    vz    = [0, 103, 206, 401]
    #zor   = vz[0];
    lenx  = 300
    mystp = [[4, 68, 132, 196],
            [4, 103, 167, 231],
            [4, 68, 132, 196],
            [4, 103, 167, 231]]
    mystp = np.array(mystp) - lenx/2
    # mwstp = [[64, 64, 64, 99],   # effective strip width
    #          [99, 64, 64, 64],
    #          [64, 64, 64, 99],
    #          [99, 64, 64, 64]]
    # Tentative uncertainties
    # sts = 0.75    # t0 uncertainty (it was 0.3)
    # std = 0.05  # tx uncertainty (it was 0.085)
    #
    #%%    
    npar = 6
    nvar = 3
    fdat = []
    vdat = np.zeros(nvar)
    mfit = np.zeros(npar)
    mch2 = np.zeros(nvar)
    vchi = []   # chi2 values
    mcha = []   # Charges
    vchr = []
    vc2y = []
    vcts = []
    vctd = []
    vcsf = []  # chi2 survival function
    m_res_ip = []
    #%%
    via  = [1,2,3,4]                      # indices array
    via  = np.expand_dims(via, axis=1)
    vz   = np.expand_dims(vz, axis=1)
    #%%
    print("Reading the datafile...")
    with open(filename,'rb') as file:
        while True:
            try:
                matrix = np.load(file)
                fdat.append(matrix)
            except ValueError:
                break
    ntrk  = len(fdat)
    #%%
    #
    #%%
    #
    dist = 10
    nft  = 0     # nb. of fitted tracks
    
    if Limit and limit_number < ntrk:
        ntrk = limit_number  # ===================================== Nb. of tracks to be fitted
    
    print("-----------------------------")
    print(f"{ntrk} events to be fitted")
    print("-----------------------------")
    
    n_residuals = 0
    for it in range(ntrk):
        if it % 1000 == 0: print(f"Trace {it}")
        #mdat = fdat[it]
        #
        adat = fdat[it]                        # input data array
        #mdat = adat
        ###
        mdata = np.concatenate((vz, adat), axis=1)
        mdata = np.concatenate((via, mdata), axis=1)
        mdat  = mdata
        #
        mdat[:,4] = 30
        mdat[:,6] = 0.3
        mdat[:,8] = 0.085
        #
        # mdat[:,4] = 1
        # mdat[:,6] = 1
        # mdat[:,8] = 1
        #  
        #
        for ip in range(nplan):      # we delete not fired planes
            if np.any(mdata[ip][:] < -9999 ):
                mdat = np.delete(mdata, ip, axis=0)  # data to be analyzed
    #           continue
        #
        #if np.any(mdat < -9999): continue
        #
        nfip = 0
        ndat = 0
        nfip = len(mdat)
        #print('n filred planes', nfip)
        #    
        if nfip > 1:
            #
            vs  = np.asarray([0,0,0,0,0,sc])
            mk  = np.zeros([npar, npar])
            # mki = np.zeros([npar, npar])
            va  = np.zeros(npar)
            istp = 0   # nb. of fitting steps
            #
            dist = d0
            while dist>cocut:
                for ip in range(nfip):
                    #
                    zi  = mdat[ip][1]
                    yst = mdat[ip][3]
                    sy  = mdat[ip][4]
                    ts  = mdat[ip][5]
                    sts = mdat[ip][6]
                    td  = mdat[ip][7]
                    std = mdat[ip][8]
                    #
                    vdat = [yst, ts, td]
                    vsig = [sy, sts, std]
                    #
                    mk = mk + fmkx(nvar, npar, vs, vsig, ss, zi)
                    va = va + fvax(nvar, npar, vs, vdat, vsig, lenx, ss, zi)
                #
                istp = istp + 1
                merr = linalg.inv(mk)     # Error matrix
                vs0 = vs
                #
                vs  = merr @ va          # sEa equation
                #
                dist = fmahd(npar, vs, vs0, merr)
                # print("***** ", istp, dist)
            dist = 10
            nft = nft + 1  # nb. of fitted tracks
            vsf = vs       # final saeta
    #%%
        '''
        for ip in range(nfip):
            zi  = mdat[ip][1]
            xi = vs[0] + vs[1] * zi
            yi = vs[2] + vs[3] * zi
            ti = vs[4] + vs[5] * zi
            print ('n_fired_planes :', nfip)
            print ('- zi, xi, yi, ti: ', zi, xi, yi, ti)
        print()
        '''
        
    #%%        
        # Chi2 -------------------------------------------------------
        chi2 = fs2(nvar, npar, vs, vdat, vsig, lenx, ss, zi)
        vchi.append(chi2)
        # Charges --------------------
        # all_charges_and_s = [ fdat[it][:,7], fdat[it][:,8] ]
        all_charges_and_s = mdat[:,9]
        mcha.append(all_charges_and_s)
        #
        # Calculating the residuals
        ch2_res  = 0
        ch2_yst  = 0
        ch2_tsum = 0
        ch2_tdif = 0
        ndat     = 0
        #
        for ip in range(nfip):
            #print('---', ip)    
            ndat = ndat + nvar
            #
            zi  = mdat[ip][1]
            yst = mdat[ip][3]
            sy  = mdat[ip][4]
            ts  = mdat[ip][5]
            sts = mdat[ip][6]
            td  = mdat[ip][7]
            std = mdat[ip][8]
            #
            vdat = [yst, ts, td]
            vsig = [sy, sts, std]
            #
            # print("EXTERNAL residues:")
            vres       = fres(vs, vdat, vsig, lenx, ss, zi)
            ch2_yst   = ch2_yst  + vres[0]**2
            ch2_tsum  = ch2_tsum + vres[1]**2
            ch2_tdif  = ch2_tdif + vres[2]**2
            ch2_res = ch2_res + ch2_yst + ch2_tsum + ch2_tdif
            #print('ip, vres: ',ip, ' ', vres)
        ndf  = ndat - npar    # number of degrees of freedom; it was ndat - npar
        #
        vc2y.append(ch2_yst)
        vcts.append(ch2_tsum)
        vctd.append(ch2_tdif)
        vchr.append(ch2_res)
        ch2_sfunction = stats.chi2.sf(ch2_res, ndf) # chi2 goodnes, survival function = 1 - F(chi2, ndf)
        vcsf.append(ch2_sfunction)
        #print(mfit)
        #print('*',vs)
        vs = np.array(vs)
        mfit = np.vstack((mfit,vs))
        # mfit = np.append(mfit,vs)
        vch2 = np.array([ch2_yst, ch2_tsum, ch2_tdif])
        mch2 = np.vstack((mch2, vch2))
            
    #%%    --------------------------------  Residual analysis with 4-plane tracks
        if (nfip < nplan):  continue
        n_residuals = n_residuals + 1
        # print('--------------- n_residuals: ', n_residuals)   
        #
        mres_ip = np.zeros(nvar)
        vs  = vsf  # We start with the previous 4-planes fit
        #
        for i_plane in range(nplan):
            
            # We hide every plane and make a TT fit in the 3 remaining planes
            y_strip_ref = mdat[i_plane][3]
            t_sum_ref = mdat[i_plane][5]
            t_dif_ref = mdat[i_plane][7]
            vdat_ip = [ y_strip_ref, t_sum_ref, t_dif_ref]
            # print(vdat_ip)
            
            mdat_short  = np.delete(mdat, i_plane, axis=0)  # mdat reduced
            
            z_ref  = mdat[i_plane][1]       # We take as reference the hidden plane
            mk     = np.zeros([npar, npar])
            va     = np.zeros(npar)
            #
            dist = d0
            while dist>cocut:
                for ifp in range(len(mdat_short)):   # loop on fired planes
                    #
                    zi       = mdat_short[ifp][1] - z_ref
                    # zi       = mdat_short[ifp][1]
                    y_strip  = mdat_short[ifp][3]
                    sig_y    = mdat_short[ifp][4]
                    t_sum    = mdat_short[ifp][5]
                    sig_tsum = mdat_short[ifp][6]
                    t_dif    = mdat_short[ifp][7]
                    sig_tdif = mdat_short[ifp][8]
                    #
                    vdat = [y_strip, t_sum, t_dif]
                    vsig = [sig_y, sig_tsum, sig_tdif]
                    #
                    mk = mk + fmkx(nvar, npar, vs, vsig, ss, zi)
                    va = va + fvax(nvar, npar, vs, vdat, vsig, lenx, ss, zi)
                #
                merr = linalg.inv(mk)     # Error matrix
                vs0 = vs
                #
                vs  = merr @ va          # ---------------------  sEa equation
                #
                dist = fmahd(npar, vs, vs0, merr)
                
            #
            #  Residuals analysis
            #
            v_it_is = [i_plane+1, mdat[i_plane][2]]
            # print("Internal residues:")
            v_res   = fres(vs, vdat_ip, vsig, lenx, ss, zi)
            v_res_ip = v_it_is + v_res
            # print('--------- v_res_ip: ', v_res_ip)
            # #vres_ip  = np.array(vres_ip)
            # #print('--------- vres: ', vres)
            # print()
            
            m_res_ip.append(v_res_ip)
            
            # '''
            # print('--- vs_3 planes: ', vs)
            # print('--- i_plane, res(x,y,t): ', i_plane,' - ', vres_ip)
            # print()
            # '''
                    
    #%%
    #print('---', mfit)
    mfit     = np.delete(mfit,0,0)
    mch2     = np.delete(mch2,0,0)
    mres_ip  = np.delete(mres_ip,0,0)
    #print('===', mfit)
    return mfit, vchi, vcsf, mcha, m_res_ip