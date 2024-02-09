import numpy as np

def window1D(ki,kN):
    Wi = (kN*np.sin(0.5*np.pi*ki/kN)/(0.5*np.pi*ki))
    zero = np.where(ki == 0)[0]
    if len(zero) > 0:
        Wi[zero] = 1.
    return Wi

def window3Dsq(kx,ky,kz,kN):
    """
    Jing et al. (2005) equation 18, assume p=3 since we use TSC (rather than the original source code)
    """
    return window1D(kx,kN)**3 * window1D(ky,kN)**3 * window1D(kz,kN)**3

realk= np.random.rand(100,100,100)
realp= np.random.rand(100,100,100)
silent=False

def dealias(k,p,ngrid,L,conv,nsize,dircut,realk=realk,realp=realp,silent=silent):
    """
    De-alias the shot noise subtracted power spectrum according to Jing 2005.
    """
    if 'L' not in locals():
        L = 400.
    if 'ngrid' not in locals():
        ngrid = 2*L**(1+np.round(np.log(np.max(k)*L/(2.*np.pi)/np.sqrt(3.))/np.log(2.)))
    if 'conv' not in locals():
        conv = 0.01 #convergence limit for alpha
    if 'nsize' not in locals():
        nsize = 5 #extent of Dirac grid, formally nsize=infinity but contributions drop off exponentially
    if 'dircut' not in locals():
        dircut = 100 #how many different k directions are averaged over (formally all of them, but using 100 is usually within 1%)

    if ngrid < 1620:
        ngrid = np.uint64(ngrid)
        ngrid2 = int(ngrid // 2)
        kx = np.indices((ngrid2, ngrid2, ngrid2), dtype=np.uint64)
    else:
        ngrid = np.int64(ngrid)
        ngrid2 = int(ngrid // 2)
        kx = np.indices((ngrid2, ngrid2, ngrid2), dtype=np.int64)

    ky = np.copy(kx)
    kz = np.copy(kx) 
    kx = kx % ngrid2
    ky = (ky % (ngrid2*ngrid2))/ngrid2
    kz = kz/(ngrid2*ngrid2)
    kk = np.round(np.sqrt(kx*kx+ky*ky+kz*kz))

    kr = np.round(k*L/(2.*np.pi))
    nk = kr.size
    kN = np.copy(ngrid2)

    ok = np.where((ky >= kx)&(kz >= ky)&(kz > 0))[0]
    islope = np.where((kr >= ngrid/(10.**0.1))&(kr <= ngrid2*10.**0.1)) #estimate slope with 0.1dex offset 

    nn = np.where((kx <= 2*nsize)&(ky <= 2*nsize)&(kz <= 2*nsize)) #re-use k grid to save memory

    h, _ = np.histogram(kk[ok], bins=np.arange(nk + 2))

    dalpha = 100. 

    #alpha=(mpfitexpr('P[0]+P[1]*X',alog10(kr[islope]),alog10(p[islope]),1d-2,[2D,-2D],/quiet))[1]

    C2 = np.zeros(nk) + (window1D(kr, kN) ** 2).flatten() #added a flatten here since the shapes did not match

    if 'realk' in locals() and 'realp' in locals():
        realkr = np.round(realk*L/(2*np.pi))
        C2real = 10.**(np.interp(np.log10(p),np.log10(kr),np.log10(kN)))/10.**(np.interp(np.log10(realp),np.log10(realkr),np.log10(kN)))
        if 'silent' not in locals():
            print('Aliased P(kN):'+str(10.**(np.interp(np.log10(p),np.log10(kr),np.log10(kN)))))
            print('Real P(kN):'+str(10.**(np.interp(np.log10(realp),np.log10(realkr),np.log10(kN)))))
            print('Actual factor:'+str(C2real))
    
    ave = 0.
    count = 0.
    jj = np.arange(h[kN])
    if h[kN] > dircut:
        rr = np.random.rand(h[kN])
        sort = np.argsort(rr)
        rr_sort = rr[sort]
        jj = jj[rr_sort[0:dircut]]

    for j in range(0,min(h[kN],dircut)-1):
        index = ok[sort[sort[kN]+jj[j]]]
        kxtmp = kx[index] + 2*kN*(kx[nn]-nsize)
        kytmp = ky[index] + 2*kN*(ky[nn]-nsize)
        kztmp = kz[index] + 2*kN*(kz[nn]-nsize)
        ktmp = np.sqrt(kxtmp**2 + kytmp**2 + kztmp**2)
        W2 = window3Dsq(kxtmp,kytmp,kztmp,kN)
        term = np.sum(W2*10.**(np.interp(np.log10(realp),np.log10(realkr),np.log10(ktmp))))
        if (kx[index] != ky[index]) and (ky[index] != kz[index]):
            weight = 6
        elif (kx[index] != ky[index]) or (ky[index] != kz[index]):
            weight = 3
        else:
            weight = 1 
        if kx[index] != 0:
            weight *= 8
        elif ky[index] != 0:
            weight *= 4
        elif kz[index] != 0:
            weight *= 2
        ave += weight*term
        count += weight 
        if term == 0: 
            break

    C2 = (ave/count)/(10.**np.interp(np.log10(realp),np.log10(realkr),np.log10(kN)))
    if 'silent' not in locals():
        print('Calculated factor:' +str(C2)+' ('+str(100*C2/C2real)+'%)')
    #return C2

    while dalpha > conv: 
        for i in range(11,nk-1):
            if h[i+1] > 0:
                ave = 0.
                count = 0
                jj = np.indices(h[i+1])
                if h[i+1] > dircut:
                    rr = np.random.rand(h[kN])
                    sort = np.argsort(rr)
                    rr_sort = rr[sort]
                    jj = jj[rr_sort[0:dircut]]

                for j in range(0,min(h[kN],dircut)-1):
                    index = ok[sort[sort[kN]+jj[j]]]
                    kxtmp = kx[index] + 2*kN*(kx[nn]-nsize)
                    kytmp = ky[index] + 2*kN*(ky[nn]-nsize)
                    kztmp = kz[index] + 2*kN*(kz[nn]-nsize)
                    ktmp = np.sqrt(kxtmp**2 + kytmp**2 + kztmp**2)
                    W2 = window3Dsq(kxtmp,kytmp,kztmp,kN)
                    #term = np.sum(W2*ktmp**alpha) figure out how to get alpha 

                    if (kx[index] != ky[index]) and (ky[index] != kz[index]):
                        weight = 6
                    elif (kx[index] != ky[index]) or (ky[index] != kz[index]):
                        weight = 3
                    else:
                        weight = 1  
                    
                    if kx[index] != 0:
                        weight *= 8
                    elif ky[index] != 0:
                        weight *= 4
                    elif kz[index] != 0:
                        weight *= 2

                    ave += weight*term
                    count += weight                    
                    if term == 0: 
                        break

                #C2[i]=(ave/count)/kr[i]**alpha
                if 'silent' not in locals():
                    print('Correction factor at k='+str(kr[i])+' was '+str(C2[i]))
        pnew=p/C2
        #alphanew=(mpfitexpr('P[0]+P[1]*X',alog10(kr[islope]),alog10(pnew[islope]),1d-2,[2D,-2D],/quiet))[1]
        #dalpha=np.abs(alphanew-alpha)
        #alpha=alphanew
        #if 'silent' not in locals():
        #    print('alpha='+str(alpha)+', dalpha='+str(dalpha)+', correction factor at kN was '+str(C2[ngrid2-1]))

    
k_test = np.random.randint(2,10,size=(100,100,100))
p_test = np.random.randint(2,10,size=(100,100,100))

#Just copying the code here: 
L_test = 1000. #length of the box of FLAMINGO 
ngrid_test = 2*L_test**(1+np.round(np.log(np.max(k_test)*L_test/(2.*np.pi)/np.sqrt(3.))/np.log(2.)))
conv_test = 100
nsize_test = 5
dircut_test = 0.01

#print(dealias(k_test,p_test,L_test,ngrid_test,conv_test,nsize_test,dircut_test))

