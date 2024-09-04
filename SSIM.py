#ssim.py
import numpy as np
import scipy.signal


def ssim_index_new(img1,img2,K,win):

    M,N = img1.shape

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    C1 = (K[0]*255)**2
    C2 = (K[1]*255) ** 2
    win = win/np.sum(win)

    mu1 = scipy.signal.convolve2d(img1,win,mode='valid')
    mu2 = scipy.signal.convolve2d(img2,win,mode='valid')
    mu1_sq = np.multiply(mu1,mu1)
    mu2_sq = np.multiply(mu2,mu2)
    mu1_mu2 = np.multiply(mu1,mu2)
    sigma1_sq = scipy.signal.convolve2d(np.multiply(img1,img1),win,mode='valid') - mu1_sq
    sigma2_sq = scipy.signal.convolve2d(np.multiply(img2, img2), win, mode='valid') - mu2_sq
    img12 = np.multiply(img1, img2)
    sigma12 = scipy.signal.convolve2d(np.multiply(img1, img2), win, mode='valid') - mu1_mu2

    if(C1 > 0 and C2>0):
        ssim1 =2*sigma12 + C2
        ssim_map = np.divide(np.multiply((2*mu1_mu2 + C1),(2*sigma12 + C2)),np.multiply((mu1_sq+mu2_sq+C1),(sigma1_sq+sigma2_sq+C2)))
        cs_map = np.divide((2*sigma12 + C2),(sigma1_sq + sigma2_sq + C2))
    else:
        numerator1 = 2*mu1_mu2 + C1
        numerator2 = 2*sigma12 + C2
        denominator1 = mu1_sq + mu2_sq +C1
        denominator2 = sigma1_sq + sigma2_sq +C2

        ssim_map = np.ones(mu1.shape)
        index = np.multiply(denominator1,denominator2)
        #如果index是真，就赋值，是假就原值
        n,m = mu1.shape
        for i in range(n):
            for j in range(m):
                if(index[i][j] > 0):
                    ssim_map[i][j] = numerator1[i][j]*numerator2[i][j]/denominator1[i][j]*denominator2[i][j]
                else:
                    ssim_map[i][j] = ssim_map[i][j]
        for i in range(n):
            for j in range(m):
                if((denominator1[i][j] != 0)and(denominator2[i][j] == 0)):
                    ssim_map[i][j] = numerator1[i][j]/denominator1[i][j]
                else:
                    ssim_map[i][j] = ssim_map[i][j]

        cs_map = np.ones(mu1.shape)
        for i in range(n):
            for j in range(m):
                if(denominator2[i][j] > 0):
                    cs_map[i][j] = numerator2[i][j]/denominator2[i][j]
                else:
                    cs_map[i][j] = cs_map[i][j]


    mssim = np.mean(ssim_map)
    mcs = np.mean(cs_map)

    return  mssim,mcs

