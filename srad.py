import numpy as np

class SRAD3D():
    def __init__(self, DegreeOfSmoothing=0.5 ,NumIterations=10,rect=(128,128,128,64,64,64)):
        self.DegreeOfSmoothing = DegreeOfSmoothing
        self.NumIterations = NumIterations
        self.rect=rect

    def setDegreeOfSmoothing(self, DegreeOfSmoothing):
        self.DegreeOfSmoothing = DegreeOfSmoothing

    def setNumIterations(self, NumIterations):
        self.NumIterations = NumIterations
    def setRect(self, rect):
        self.rect = rect

    def run(self, img):
        img=np.abs(img)
        
        size_x,size_y,size_z = np.shape(img)
        iN = np.concatenate((np.array([0]),np.arange(0,size_x-1)))
        iS = np.concatenate((np.arange(1,size_x),np.array([size_x-1])))
        jW = jW = np.concatenate((np.array([0]),np.arange(0,size_y-1)))
        jE = np.concatenate((np.arange(1,size_y),np.array([size_y-1])))
        kU = np.concatenate((np.array([0]),np.arange(0,size_z-1)))
        kD = np.concatenate((np.arange(1,size_z),np.array([size_z-1])))

        ''' prepare cropping indices '''
        xl=self.rect[0]
        xu=self.rect[0]+self.rect[3]
        yl=self.rect[1]
        yu=self.rect[1]+self.rect[4]
        zl=self.rect[2]
        zu=self.rect[2]+self.rect[5]
        for i in range(self.NumIterations):
            ''' Display progress '''
            print("# INFO : Iteration : ",i)
            ''' speckle scale function '''
            Iuniform = img[xl:xu,yl:yu,zl:zu]
            q0_squared = (np.std(Iuniform)/np.mean(Iuniform))**2

            ''' differences '''
            dN=img[tuple(iN),:,:]-img
            dS=img[tuple(iS),:,:]-img
            dW=img[:,tuple(jW),:]-img
            dE=img[:,tuple(jE),:]-img
            dU=img[:,:,tuple(kU)]-img
            dD=img[:,:,tuple(kD)]-img

            ''' normalized discrete gradient magnitude squared (equ 52,53) '''
            G2 = (dN**2 + dS**2 + dW**2 + dE**2 + dU**2 + dD**2) / (img**2 + np.finfo(float).eps)

            ''' normalized discrete laplacian (equ 54) '''
            L = (dN + dS + dW + dE + dU + dD) / (img + np.finfo(float).eps)

            ''' ICOV (equ 31/35) '''
            ita = 6
            num = ((1/3)*G2) - ((1/ita**2)*(L**2))
            den = (1 + ((1/ita)*L))**2
            q_squared = num / (den +  np.finfo(float).eps)

            ''' diffusion coefficent (equ 33) '''
            den = (q_squared - q0_squared) / (q0_squared *(1 + q0_squared) + np.finfo(float).eps)
            c = 1 / (1 + den)
            cS = c[tuple(iS),:,:]
            cE = c[:,tuple(jE),:]
            cD = c[:,:,tuple(kD)]

            ''' divergence equ 58 '''
            D = (cS*dS) + (c*dN) + (cE*dE) + (c*dW) + (cD*dD) + (c*dU)

            ''' update (equ 61) '''
            img = img + (self.DegreeOfSmoothing/ita)*D

        ''' retrun the results '''
        return img
