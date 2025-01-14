#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  6 paramter version
"""
Created on Fri Mar  8 10:28:00 2024

@author: cayesoneira
0: Version JAG
1: Version Caye
2: Version JAG
3: Version Caye
4: Version JAG
ap3: Version Caye
ap3_2: Version JAG
"""
    
def tt_nico(filename):
    Limit = False
    limit_number =  2000
    #
    import numpy as np
    import scipy.linalg as linalg
    import scipy.stats as stats
    from math import sqrt
    import os
    
    # datafile = "timtrack_data_ypos_cal_pos_cal_time.bin"
    datafile = filename
    sigmafile = "uncertainties_per_strip.txt"
    
    print("-----------------------------")
    if os.path.exists(sigmafile):
      print(f"The file {sigmafile} exists in the working directory.")
      use_uncertainties = True
    else:
      print(f"The file {sigmafile} does not exist in the working directory.")
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
    def find_s2(nvar, npar, vs, vdat, vsig, lenx, ss, zi):
      mk = fmkx(nvar, npar, vs, vsig, ss, zi)
      va = fvax(nvar, npar, vs, vdat, vsig, lenx, ss, zi)
      vm = fvmx(nvar, vs, lenx, ss, zi)
      mg = fmgx(nvar, npar, vs, ss, zi)
      mw = fmwx(nvar, vsig)
      vg = vm - mg @ vs
      vdmg = vdat - vg
      sk = vs.transpose() @ mk @ vs         # mk contribution to s2
      sa = vs.transpose() @ va              # va contribution to s2
      s0 = vdmg.transpose() @ mw @ vdmg     # free term
      s2 = sk - 2*sa + s0
      return s2
    def fmahd(npar, vin1, vin2, merr): # Mahalanobis distance
      vdif  = np.subtract(vin1,vin2)
      vdsq  = np.power(vdif,2)
      verr  = np.diag(merr,0)
      vsig  = np.divide(vdsq,verr)
      dist  = np.sqrt(np.sum(vsig))
      return dist
    def find_residuals(vs, vdat, vsig, lenx, ss, zi):  # Residual array
    # Find fit residuals    
      sy  = vsig[0]; sts = vsig[1]; std = vsig[2]
      X0  = vs[0]; XP = vs[1]; Y0 = vs[2]; YP = vs[3]; T0 = vs[4]; S0 = vs[5]
      kz  = sqrt(1 + XP*XP + YP*YP)
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
    
      DeltaX = abs ( (tsdat - ( T0 + S0 * kz * zi ) ) / 0.5 / ss - lenx)
      # Residuals array
      vres = [yr, tsr, tdr, DeltaX]
      return vres
    #%%#   A few constants
    #  
    vc    = 300 #mm/ns
    sc    = 1/vc
    vp    = 2/3 * vc  # velocity of the signal in the strip
    ss    = 1/vp
    sq12i = 1/np.sqrt(12)
    #
    cocut = 1  # convergence cut
    d0    = 10 # initial value of the convergence parameter 
    islim = 5  # fit step limit
    #%% Detector layout
    nplan = 4
    lenx  = 300
    vz    = [0, 103, 206, 401]
    vz    = np.expand_dims(vz, axis=1)
    #zor   = vz[0];
    mystp = [[4, 68, 132, 196],
          [4, 103, 167, 231],
          [4, 68, 132, 196],
          [4, 103, 167, 231]]
    mystp = np.array(mystp) - lenx/2   # the coordinate origin is at the middle of the detector
    mwstp = [[64, 64, 64, 99],   # effective strip width
           [99, 64, 64, 64],
           [64, 64, 64, 99],
           [99, 64, 64, 64]]
    # Time uncertainties
    sts = 0.3   # t0 uncertainty (it was 0.3)
    std = 0.093 # tx uncertainty (it was 0.085)
    #
    #%%  Fit parameters    
    npar = 6
    nvar = 3
    #
    fdat = []
    mcha = []        # Charges
    mcha_slew = []   # Charges
    mDeltaY_both = []
    mres_final = []
    vchi2   = []        # tt chi2 
    vchi2r  = []        # residual chi2
    vchi2y  = []
    vchi2s  = []
    vchi2d  = []
    vfsurf  = []  # fit chi2 survival function
    vrsurf  = []  # residual chi2 survival function
    vdat  = np.zeros(nvar)
    msfit = np.zeros(npar)
    mchi2 = np.zeros(nvar + 4)
    mres_ip = np.zeros(nvar + 4)
    #
    via  = [1,2,3,4]                       # strip indices array
    via  = np.expand_dims(via, axis=1)
    #%%   Read datafiles
    #
    print("-----------------------------")
    print("Reading the datafile...")
    i = 0
    with open(datafile, 'rb') as file:
      while (not Limit or i < limit_number):
          try:
              matrix = np.load(file)
              fdat.append(matrix)
              i += 1
          except ValueError:
              break
    ntrk  = len(fdat)     # number of events in the fdat file
    #
    #%%  Fit
    #
    n_ftracks  = 0     # nb. of fitted tracks
    if Limit and limit_number < ntrk:
      ntrk = limit_number  # ========================= Nb. of tracks to be fitted
    
    print("-----------------------------")
    print(f"{ntrk} events to be fitted")
    print("-----------------------------")
    
    n_residuals = 0       # residuals count
    for it in range(ntrk):
      if it % 1000 == 0: print(f"Trace {it}")
      #
      adat = fdat[it]                        
      mdata = adat
      # We add plane index and z-coordinate to the data matrix
      mdata = np.concatenate((vz, adat), axis=1)
      mdata = np.concatenate((via, mdata), axis=1)
      mdat = mdata
      #
      # mdat: [iplane, zplane, istrip, ycoor, sycoor, ts, sts, td, std, q, sq]
      #
      for ip in range(nplan):                       # we ignore not fired planes
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
      n_fplanes = 0
      ndat = 0
      n_fplanes = len(mdat)
      #print('n filred planes', n_fplanes)
      #   
      if n_fplanes > 1:
          #
          vs  = np.asarray([0,0,0,0,0,sc])
          mk  = np.zeros([npar, npar])
          va  = np.zeros(npar)
          #
          i_step = 0   # nb. of fitting steps
          dist = d0
          while dist>cocut:
              for ip in range(n_fplanes):
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
              i_step = i_step + 1
              #
              merr = linalg.inv(mk)    # --------------------------    Error matrix
              #
              vs0 = vs
              vs  = merr @ va          # --------------------------    sEa equation
              #
              dist = fmahd(npar, vs, vs0, merr)
              if i_step > islim: n_fplanes = 0;  continue
              # 
          dist = 10
          n_ftracks = n_ftracks + 1  # nb. of fitted tracks
          vsf = vs       # final saeta  
            #all_charges_and_s = mdat[:,9:13] # --------------------------------   ?
      vs    = np.array(vs)
      msfit = np.vstack((msfit,vs))
      #
      all_charges_and_s = mdat[:,9:10]
      mcha.append(all_charges_and_s)
      # mDeltaY.append(mdat[:,12])
    #%%   Chi2 
    # TT fit chi2
      fchi2 = find_s2(nvar, npar, vs, vdat, vsig, lenx, ss, zi)
    # Residuals chi2
      chi2_res  = 0    # residuals chi2
      chi2_yst  = 0    # y coordinate chi2
      chi2_tsum = 0    # tsum chi2
      chi2_tdif = 0    # tidf chi2
      ndat      = 0
      rchi2 = 0  # Initialize rchi2 before the loop
      for ip in range(n_fplanes):
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
          vres       = find_residuals(vs, vdat, vsig, lenx, ss, zi)
          chi2_yst   = chi2_yst  + vres[0]**2
          chi2_tsum  = chi2_tsum + vres[1]**2
          chi2_tdif  = chi2_tdif + vres[2]**2
          rchi2      = chi2_yst + chi2_tsum + chi2_tdif # residual chi2
          #print('--- ip, vres: ',ip, ' ', vres)
      vchi2.append(fchi2)
      vchi2y.append(chi2_yst)
      vchi2s.append(chi2_tsum)
      vchi2d.append(chi2_tdif)
      vchi2r.append(chi2_res)
      # chi2 goodness, survival function = 1 - F(chi2, ndf)
      ndf  = ndat - npar    # number of degrees of freedom 
      fsurf = stats.chi2.sf(fchi2, ndf) 
      vfsurf.append(fsurf)
      rsurf = stats.chi2.sf(rchi2, ndf) 
      vrsurf.append(rsurf)
      #
      allch2 = np.array([fchi2, fsurf, rchi2, rsurf, chi2_yst, chi2_tsum, chi2_tdif])
      mchi2  = np.vstack((mchi2, allch2))
    #%%    --------------------------------  Residual analysis with 4-plane tracks
      if (n_fplanes < nplan):  continue
      n_residuals = n_residuals + 1 
      #
      for i_plane in range(nplan):   
          # We hide every plane and make a TT fit in the 3 remaining planes
          smdat  = np.delete(mdat, i_plane, axis=0)  # short mdat
          #
          z_ref   = mdat[i_plane][1]       
          isp_ref = mdat[i_plane][2]
          ys_ref  = mdat[i_plane][3]
          ts_ref  = mdat[i_plane][5]
          td_ref  = mdat[i_plane][7]
          vd_ref  = [ys_ref, ts_ref, td_ref]
          vs_ref  = [mdat[i_plane][4], mdat[i_plane][6], mdat[i_plane][8]]
          #
          vs   = vsf  # We start with the previous 4-planes fit
          mk   = np.zeros([npar, npar])
          va   = np.zeros(npar)
          ist3 = 0
          dist = d0
          while dist > cocut:
              for ip in range(len(smdat)):   # loop on fired planes
                  #
                  zi  = smdat[ip][1] - z_ref  #  z-coords are refered to the hidden plane
                  yst = smdat[ip][3]
                  sy  = smdat[ip][4]
                  ts  = smdat[ip][5] 
                  sts = smdat[ip][6]
                  td  = smdat[ip][7]
                  std = smdat[ip][8]
                  #
                  vdat = [yst, ts, td]
                  vsig = [sy, sts, std]
                  #
                  mk = mk + fmkx(nvar, npar, vs, vsig, ss, zi)
                  va = va + fvax(nvar, npar, vs, vdat, vsig, lenx, ss, zi)
              #
              ist3 = ist3 + 1
              #
              merr = linalg.inv(mk)     # --------------------  Error matrix
              #          
              vs0 = vs
              vs  = merr @ va           # ---------------------  sEa equation
              #
              dist = fmahd(npar, vs, vs0, merr)
              if ist3 > islim: n_fplanes = 0;  continue 
          #
          vtrack  = [it+1, i_plane+1, isp_ref ]
          print(vtrack)
          # Residuals: vres =[yr, tsr, tdr, DeltaX]
          vres    = find_residuals(vs, vd_ref, vs_ref, lenx, ss, 0)
          vres_ip = np.array(vtrack + vres)
          mres_ip = np.vstack((mres_ip, vres_ip))
          v_res_ip = vtrack + vres + [mdat[np.int(i_plane), 14]]
          # print(vtrack)
          # print(vres)
          # print([mdat[np.int(i_plane), 14]])
          # print(v_res_ip)
      
      v_res_ip  = np.array(v_res_ip)
      mres_final.append(v_res_ip)
          
      #m_res_ip.append(v_res_ip)
      mcha_slew.append(all_charges_and_s)
    #
    mfit   = np.delete(msfit,0,0)
    mchi2  = np.delete(mchi2,0,0)
    #
    return mfit, vchi2, vrsurf, mcha, mcha_slew, mres_final