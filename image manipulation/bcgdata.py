import numpy as np;

def read_bcg_data ():
    mmv,emmv,mscmv,emscmv,mssmv,emssmv=np.loadtxt("bcgmasses_new.dat",usecols=(2,3,7,8,14,15),unpack=True);

    # convert from diet Salpeter to the Chabrier
    mscmv = mscmv * 0.807
    emscmv = emscmv * 0.807
    mssmv = mssmv * 0.807
    emssmv = emssmv * 0.807

    mmv = np.delete(mmv,[9,10])
    emmv = np.delete(emmv,[9,10])

    mssmv[6] = mssmv[6] + mscmv[10]
    emssmv[6] = np.sqrt(emssmv[6]**2+emscmv[10]**2)
    mssmv[4] = mssmv[4] + mscmv[9]
    emssmv[4] = np.sqrt(emssmv[4]**2+emscmv[9]**2)

    mscmv = np.delete(mscmv,[9,10])
    mssmv = np.delete(mssmv,[9,10])
    emscmv = np.delete(emscmv,[9,10])
    emssmv = np.delete(emssmv,[9,10])

    mstmv = mscmv + mssmv
    emstmv = np.sqrt(emscmv**2+emssmv**2)
    #emstmv = 0.5*(np.log10(mstmv+emstmv)-np.log10(mstmv-emstmv))
    print mmv, mstmv, emstmv
    emstmv = 0.434 * emstmv/mstmv
    mstmv = np.log10(mstmv)

    #emmv = 0.5*(np.log10(mmv+emmv)-np.log10(mmv-emmv))
    # compute log error as dlog y~dy/y*0.434 assuming errors are small
    emmv = 0.434*emmv/mmv
    mmv = np.log10(mmv)

    #emscmv = 0.5*(np.log10(mscmv+emscmv)-np.log10(mscmv-emscmv))
    emscmv = 0.434*emscmv/mscmv
    mscmv = np.log10(mscmv)
    #emssmv = 0.5*(np.log10(mssmv+emssmv)-np.log10(mssmv-emssmv))
    emssmv = 0.434*emssmv/mssmv
    mssmv = np.log10(mssmv)
       
    return mmv, emmv, mstmv, emstmv