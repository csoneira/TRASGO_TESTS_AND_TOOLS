# !/ur/bi/envpythn3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:19:05 2024

@author: cayesoneira
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:19:05 2024

@author: cayesoneira
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:28:00 2024

@author: cayesoneira
0: Version JAG
1: Version Caye
2: Version JAG
3: Version Caye
4: Version JAG
"""
def tt_nico(filename):
    
    # filename = "timtrack_data_ypos_cal_pos_cal_time.bin"
    
    strip_length = 300
    
    Limit = True
    limit_number =  4000
    import numpy as np
    import scipy.linalg as linalg
    import scipy.stats as stats
    from math import sqrt
    
    import os
    file_name = "uncertainties_per_strip.txt"
    
    print("-----------------------------")
    if os.path.exists(file_name):
        print(f"The file {file_name} exists in the working directory.")
        use_uncertainties = True
    else:
        print(f"The file {file_name} does not exist in the working directory.")
        use_uncertainties = False
    
    # Uncertainties
    if use_uncertainties:
        flattened_matrix_loaded = np.loadtxt('uncertainties_per_strip.txt')
        uncertainties_per_strip = flattened_matrix_loaded.reshape((4, 4, 3))
    
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
        xfit  = X0 + XP * zi
        yfit  = Y0 + YP * zi
        tffit = T0 + S0 * kz * zi + (lenx/2 + xfit) * ss
        tbfit = T0 + S0 * kz * zi + (lenx/2 - xfit) * ss
        tsfit = 0.5 * (tffit + tbfit)
        tdfit = 0.5 * (tffit - tbfit)
        #  Data
        ydat  = vdat[0]
        tsdat = vdat[1]
        tddat = vdat[2]
        # Residuals
        # yr   = (yfit   - ydat)/sy
        # tsr  = (tsfit  - tsdat)/sts
        # tdr  = (tdfit  - tddat)/std
        yr   = (yfit   - ydat)
        tsr  = (tsfit  - tsdat)
        tdr  = (tdfit  - tddat)
        # DeltaX = abs( (tsdat - ( T0 + S0 * kz * zi ) ) / 0.5 / ss - strip_length)
        # DeltaX = abs( strip_length - abs( tsdat - ( T0 + S0 * kz * zi ) ) / 0.5 / ss )
        DeltaX = abs ( (tsdat - ( T0 + S0 * kz * zi ) ) / 0.5 / ss - strip_length )
        # Residuals array
        vres = [yr, tsr, tdr, DeltaX]
        return vres
    #%%#   A few constants
    vc    = 300 #mm/ns
    sc    = 1/vc
    vp    = 2/3 * vc  # velocity of the signal in the strip
    ss    = 1/vp
    sq12i = 1/np.sqrt(12)
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
    mwstp = [[64, 64, 64, 99],   # effective strip width
             [99, 64, 64, 64],
             [64, 64, 64, 99],
             [99, 64, 64, 64]]
    # Tentative uncertainties
    sts = 0.3   # t0 uncertainty (it was 0.3)
    std = 0.093 # tx uncertainty (it was 0.085)
    #
    #%%    
    npar = 6
    nvar = 3
    fdat = []
    vdat = np.zeros(nvar)
    mfit = np.zeros(npar)
    mch2 = np.zeros(nvar)
    # m_res_ip = np.zeros(nvar+3)
    m_res_ip = []
    vchi = []   # chi2 values
    mcha = []   # Charges
    mcha_slew = []   # Charges
    mDeltaY_both = []
    vchr = []
    vc2y = []
    vcts = []
    vctd = []
    vcsf = []  # chi2 survival function
    #%%
    via  = [1,2,3,4]                      # indices array
    via  = np.expand_dims(via, axis=1)
    vz   = np.expand_dims(vz, axis=1)
    #%%
    print("-----------------------------")
    print("Reading the datafile...")
    i = 0
    with open(filename, 'rb') as file:
        while (not Limit or i < limit_number):
            try:
                matrix = np.load(file)
                fdat.append(matrix)
                i += 1
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
        #
        adat = fdat[it]                        # input data array
        mdata = adat
        mdat = mdata
        #
        for ip in range(nplan):      # we delete not fired planes
            if np.any(mdata[ip][:] < -9999 ):
                mdat = np.delete(mdata, ip, axis=0)  # data to be analyzed
        #
        #
        # We asign the correct uncertainties to the data        
        for ip in range(len(mdat)):   # 
            ifp = np.int(mdat[ip][0]) - 1    # index of fired plane
            ifs = np.int(mdat[ip][2]) - 1    # index of fired strip
            
            if use_uncertainties:
                # mdat[ip][4] = mwstp[ifp][ifs] * sq12i
                mdat[ip][4] = uncertainties_per_strip[ifp,ifs,0]
                mdat[ip][6] = uncertainties_per_strip[ifp,ifs,1]
                mdat[ip][8] = uncertainties_per_strip[ifp,ifs,2]
            else:
                mdat[ip][4] = mwstp[ifp][ifs] * sq12i
                mdat[ip][6] = sts
                mdat[ip][8] = std
        #
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
                # print(' --loop 1---- istp: ', istp)
                merr = linalg.inv(mk)     # Error matrix
                vs0 = vs
                #
                vs  = merr @ va          # sEa equation
                #
                dist = fmahd(npar, vs, vs0, merr)
                if istp > 5:
                    nfip = 0
                    continue
                # print("*****  it, step, dist", it, istp, dist)
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
            #print ('n_fired_planes :', nfip)
            print ('- iplane, zi, xi, yi, ti: ', ip, zi, xi, yi, ti)
        print()
        '''
        
    #%%   
         
        # Chi2 -------------------------------------------------------
        chi2 = fs2(nvar, npar, vs, vdat, vsig, lenx, ss, zi)
        vchi.append(chi2)
        # Charges --------------------
        # all_charges_and_s = [ fdat[it][:,7], fdat[it][:,8] ]
        all_charges_and_s = mdat[:,9:13]
        mcha.append(all_charges_and_s)
        # mDeltaY.append(mdat[:,12])
        #
        # Calculating the residuals
        ch2_res  = 0
        ch2_yst  = 0
        ch2_tsum = 0
        ch2_tdif = 0
        ndat     = 0
        #
        #  ---  Residual analysis -------------------
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
            vres      = fres(vs, vdat, vsig, lenx, ss, zi)
            ch2_yst   = ch2_yst  + vres[0]**2
            ch2_tsum  = ch2_tsum + vres[1]**2
            ch2_tdif  = ch2_tdif + vres[2]**2
            ch2_res = ch2_res + ch2_yst + ch2_tsum + ch2_tdif
            #print('--- ip, vres: ',ip, ' ', vres)
        ndf  = ndat - npar    # number of degrees of freedom; it was ndat - npar
        #
        vc2y.append(ch2_yst)
        vcts.append(ch2_tsum)
        vctd.append(ch2_tdif)
        vchr.append(ch2_res)
        ch2_sfunction = stats.chi2.sf(ch2_res, ndf) # chi2 goodnes, survival function = 1 - F(chi2, ndf)
        vcsf.append(ch2_sfunction)
        #print(mfit)
        # print('* vres ', vres)
        # print()
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
        # mres_ip = np.zeros(nvar)
        #
        
        for i_plane in range(nplan):
            
            # We hide every plane and make a TT fit in the 3 remaining planes
            i_strip_ref = mdat[i_plane][2]
            # print(i_strip_ref)
            # print("----------------------")
            y_strip_ref = mdat[i_plane][3]
            t_sum_ref = mdat[i_plane][5]
            t_dif_ref = mdat[i_plane][7]
            vdat_ip = [ y_strip_ref, t_sum_ref, t_dif_ref]
            # print(vdat_ip)
            
            mdat_short  = np.delete(mdat, i_plane, axis=0)  # mdat reduced
            
            z_ref  = mdat[i_plane][1]       # We take as reference the hidden plane
            # print(z_ref)
            vs     = vsf  # We start with the previous 4-planes fit
            mk     = np.zeros([npar, npar])
            va     = np.zeros(npar)
            #
            ist3 = 0
            dist = d0
            while dist>cocut:
                for ip in range(len(mdat_short)):   # loop on fired planes
                    #
                    zi  = mdat_short[ip][1] - z_ref
                    yst = mdat_short[ip][3]
                    sy  = mdat_short[ip][4]
                    ts  = mdat_short[ip][5] 
                    sts = mdat_short[ip][6]
                    td  = mdat_short[ip][7]
                    std = mdat_short[ip][8]
                    #
                    vdat = [yst, ts, td]
                    vsig = [sy, sts, std]
                    #
                    mk = mk + fmkx(nvar, npar, vs, vsig, ss, zi)
                    va = va + fvax(nvar, npar, vs, vdat, vsig, lenx, ss, zi)
                #
                ist3 = ist3 + 1
                #
                merr = linalg.inv(mk)     # Error matrix
                vs0 = vs
                #
                vs  = merr @ va          # ---------------------  sEa equation
                #
                dist = fmahd(npar, vs, vs0, merr)
                # print('---- loop 3 ----- ist3, dist: ', ist3, dist)
                            #
            # --- Residual analysis
            #
            vsig = [sy, sts, std]
            #
            # print(mdat)
            # print("------------------")
            v_track  = [ it+1, i_plane+1, i_strip_ref ]
            v_res    = fres(vs, vdat_ip, vsig, lenx, ss, 0)
            v_res_ip = v_track + v_res + [mdat[np.int(i_plane), 14]]
            v_res_ip  = np.array(v_res_ip)
            
            m_res_ip.append(v_res_ip)
            mcha_slew.append(all_charges_and_s)
        # print()
    # print(m_res_ip)
    #%%
    #print('---', mfit)
    mfit     = np.delete(mfit,0,0)
    mch2     = np.delete(mch2,0,0)
    # m_res_ip = np.delete(m_res_ip,0,0)
    #print('===', mfit)
    return mfit, vchi, vcsf, mcha, mcha_slew, m_res_ip