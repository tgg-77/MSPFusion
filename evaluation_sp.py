import math
import numpy as np
import cv2
import torch
from skimage import measure,metrics
from scipy import signal
from MSSSIM import msssim
import math
from PIL import Image
from skimage.measure import shannon_entropy
from pytorch_msssim import ssim, ms_ssim
from sklearn.metrics.cluster import mutual_info_score
from tqdm import trange
from MEF_MS_SSIM.MS_SSIMc import MS_SSIMc


def spatialF(image):
    image = np.array(image)
    M = image.shape[0]
    N = image.shape[1]

    cf = 0
    rf = 0
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            dx = float(image[i, j - 1]) - float(image[i, j])
            rf += dx ** 2
            dy = float(image[i - 1, j]) - float(image[i, j])
            cf += dy ** 2

    RF = math.sqrt(rf / (M * N))
    CF = math.sqrt(cf / (M * N))
    SF = math.sqrt(RF ** 2 + CF ** 2)

    return round(SF, 3)




#求图像的灰度概率分布
def gray_possiblity(image):
    tmp = np.zeros(256,dtype='float')
    val = 0
    k = 0
    res = 0
    img = np.array(image)
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]             #像素值
            val = val.astype(int)
            tmp[val] = float(tmp[val] + 1) #该像素值的次数
            k = k+1              #总次数

    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)   #各个灰度概率
    return tmp





# 求图像的条件熵： H(x|y) = H(x,y) - H(y)
def condition_entropy(image,fuse):
    tmp = np.zeros((256,256),dtype='float')
    res = 0
    k = 0
    image = np.array(image,dtype='int')
    fuse = np.array(fuse,dtype='int')
    rows,cols = image.shape[:2]
    for i in range(len(image)):
        for j in range(len(image[i])):
            tmp[image[i][j]][fuse[i][j]] = float(tmp[image[i][j]][fuse[i][j]]+1)
            k = k+1
    for i in range(256):
        for j in range(256):
            p = tmp[i,j]/k
            if p!=0:
                res = float(res-p*math.log(p,2))
    return res - Entropy(fuse)

#求图像的信息熵 P(a)表示灰度概念 entropy = PA(a)*logPA(a)
def Entropy(image):
    tmp = []
    res = 0
    tmp = gray_possiblity(image)
    for i in range(len(tmp)):
        if(tmp[i]==0):
            res = res
        else:
            res = float(res - tmp[i]*(math.log(tmp[i],2)))
    return res

#交叉熵 求和 -(p(x)logq(x)+(1-p(x))log(1-q(x)))
def cross_entropy(image,aim):
    tmp1 = []
    tmp2 = []
    res = 0
    tmp1 = gray_possiblity(image)
    tmp2 = gray_possiblity(aim)
    for i in range(len(tmp1)):
        if(tmp1[i]!=0)and(tmp2[i]!=0):
            res = tmp1[i]*math.log(1/tmp2[i]) + res
            #res = float(res-(tmp1[i]*math.log2(tmp2[i])+(1-tmp1[i])*math.log2(1-tmp2[i])))
    return res

#求psnr 利用skimage库
def peak_signal_to_noise(true,test):
    return metrics.peak_signal_noise_ratio(true,test,data_range=255)

#求IQM 论文参考文献14
def IQM(image,fused):
    image = np.array(image,float)
    fused = np.array(fused,float)
    N = len(image)*len(image[0])
    k = 0.0
    x_mean = 0.0
    y_mean = 0.0
    for i in range(len(image)):
        for j in range(len(image[i])):
            x_mean = x_mean + image[i][j]
            y_mean = y_mean + fused[i][j]
            k = k+1
    x_mean = float(x_mean/k)
    y_mean = float(y_mean/k)
    for i in range(len(image)):
        for j in range(len(image[i])):
            xy = xy+(image[i][j]-x_mean)*(fused[i][j]-y_mean)
            x = x+(image[i][j]-x_mean)*(image[i][j]-x_mean)
            y = y+(fused[i][j]-y_mean)*(fused[i][j]-y_mean)
    xy = xy/(N-1)
    x = x/(N-1)
    y = y/(N-1)
    Q = 4*xy*x_mean*y_mean/((x*x+y*y)*(x_mean*x_mean+y_mean*y_mean))
    return Q


def QABF(stra, strb, f):
    # model parameters 模型参数
    L = 1
    Tg = 0.9994
    kg = -15
    Dg = 0.5
    Ta = 0.9879
    ka = -22
    Da = 0.8

    # Sobel Operator Sobel算子
    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
    h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)

    # if y is the response to h1 and x is the response to h3;then the intensity is sqrt(x^2+y^2) and  is arctan(y/x);
    # 如果y对应h1，x对应h2，则强度为sqrt(x^2+y^2)，方向为arctan(y/x)

    strA = stra.astype(np.float32)
    strB = strb.astype(np.float32)
    strF = f.astype(np.float32)

    # 数组旋转180度
    def flip180(arr):
        new_arr = arr.reshape(arr.size)
        new_arr = new_arr[::-1]
        new_arr = new_arr.reshape(arr.shape)
        return new_arr

    # 相当于matlab的Conv2
    def convolution(k, data):
        k = flip180(k)
        data = np.pad(data, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
        n, m = data.shape
        img_new = []
        for i in range(n - 2):
            line = []
            for j in range(m - 2):
                a = data[i:i + 3, j:j + 3]
                line.append(np.sum(np.multiply(k, a)))
            img_new.append(line)
        return np.array(img_new)

    # 用h3对strA做卷积并保留原形状得到SAx，再用h1对strA做卷积并保留原形状得到SAy
    # matlab会对图像进行补0，然后卷积核选择180度
    # gA = sqrt(SAx.^2 + SAy.^2);
    # 定义一个和SAx大小一致的矩阵并填充0定义为aA，并计算aA的值
    def getArray(img):
        SAx = convolution(h3, img)
        SAy = convolution(h1, img)
        gA = np.sqrt(np.multiply(SAx, SAx) + np.multiply(SAy, SAy))
        n, m = img.shape
        aA = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                if (SAx[i, j] == 0):
                    aA[i, j] = math.pi / 2
                else:
                    aA[i, j] = math.atan(SAy[i, j] / SAx[i, j])
        return gA, aA

    # 对strB和strF进行相同的操作
    gA, aA = getArray(strA)
    gB, aB = getArray(strB)
    gF, aF = getArray(strF)

    # the relative strength and orientation value of GAF,GBF and AAF,ABF;
    def getQabf(aA, gA, aF, gF):
        n, m = aA.shape
        GAF = np.zeros((n, m))
        AAF = np.zeros((n, m))
        QgAF = np.zeros((n, m))
        QaAF = np.zeros((n, m))
        QAF = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                if (gA[i, j] > gF[i, j]):
                    GAF[i, j] = gF[i, j] / gA[i, j]
                elif (gA[i, j] == gF[i, j]):
                    GAF[i, j] = gF[i, j]
                else:
                    GAF[i, j] = gA[i, j] / gF[i, j]
                AAF[i, j] = 1 - np.abs(aA[i, j] - aF[i, j]) / (math.pi / 2)

                QgAF[i, j] = Tg / (1 + math.exp(kg * (GAF[i, j] - Dg)))
                QaAF[i, j] = Ta / (1 + math.exp(ka * (AAF[i, j] - Da)))

                QAF[i, j] = QgAF[i, j] * QaAF[i, j]

        return QAF

    QAF = getQabf(aA, gA, aF, gF)
    QBF = getQabf(aB, gB, aF, gF)

    # 计算QABF
    deno = np.sum(gA + gB)
    nume = np.sum(np.multiply(QAF, gA) + np.multiply(QBF, gB))
    output = nume / deno
    return output




#计算图像的Qabf
def matrix_pow(m):
    for i in range(len(m)):
        for j in range(len(m[i])):
            m[i][j] = pow(m[i][j],2)
    return m
def matrix_sqrt(m):
    for i in range(len(m)):
        for j in range(len(m[i])):
            m[i][j] = math.sqrt(m[i][j])
    return m
def matrix_multi(m1,m2):
    m = np.zeros(m1.shape)
    for i in range(len(m1)):
        for j in range(len(m1[i])):
            m[i][j] = m1[i][j]*m2[i][j]
    return m
def Qabf(pA,pB,pF):
    L = 1
    Tg = 0.9994
    Kg = -15
    Dg = 0.5
    Ta = 0.9879
    Ka = -22
    Da = 0.8
    h3 = [[-1,0,1],
          [-2,0,2],
          [-1,0,1]]
    h3 = np.array(h3)
    h3 = h3.astype(np.float)
    h1 = [[1,2,1],
          [0,0,0],
          [-1,-2,-1]]
    h1 = np.array(h1)
    h1 = h1.astype(np.float)
    h2 = [[0,1,2],
          [-1,0,1],
          [-2,-1,0]]
    h2 = np.array(h2)
    h2 = h2.astype(np.float)
    SAx  = signal.convolve2d(pA,h3,'same')
    SAy = signal.convolve2d(pA,h1,'same')
    gA = matrix_pow(SAx)+matrix_pow(SAy)
    gA = matrix_sqrt(gA)
    M = SAy.shape[0]
    N = SAy.shape[1]
    aA = np.zeros(SAx.shape)
    for i in range(M):
        for j in range(N):
            if SAx[i][j]==0:
                aA[i][j] = math.pi/2
            else:
                aA[i][j] = math.atan(SAy[i][j]/SAx[i][j])

    SBx = signal.convolve2d(pB,h3,'same')
    SBy = signal.convolve2d(pB,h1,'same')
    gB = matrix_pow(SBx) + matrix_pow(SBy)
    gB = matrix_sqrt(gB)
    aB = np.zeros(SBx.shape)
    for i in range(SBx.shape[0]):
        for j in range(SBx.shape[1]):
            if SBx[i][j] == 0:
                aB[i][j] = math.pi/2
            else:

                aB[i][j] = math.atan(SBy[i][j]/SBx[i][j])
    SFx = signal.convolve2d(pF,h3,boundary='symm',mode='same')
    SFy = signal.convolve2d(pF,h1,boundary='symm',mode='same')
    gF = matrix_sqrt((matrix_pow(SFx)+matrix_pow(SFy)))
    M = SFx.shape[0]
    N = SFx.shape[1]
    aF = np.zeros(SFx.shape)
    for i in range(M):
        for j in range(N):
            if SFx[i][j] == 0:
                aF[i][j] = math.pi/2
            else:
                aF[i][j] = math.atan(SFy[i][j]/SFx[i][j])
    #the relative strength and orientation value of GAF,GBF and AAF,ABF
    GAF = np.zeros(SFx.shape)
    AAF = np.zeros(SFx.shape)
    QgAF = np.zeros(SFx.shape)
    QaAF = np.zeros(SFx.shape)
    QAF  = np.zeros(SFx.shape)
    for i in range(M):
        for j in range(N):
            if gA[i][j]>gF[i][j]:
                GAF[i][j] = gF[i][j]/gA[i][j]
            else:
                if gF[i][j] == 0:
                    GAF[i][j] = 0
                else:
                    GAF[i][j] = gA[i][j] / gF[i][j]

            AAF[i][j] = 1- abs(aA[i][j]-aF[i][j])/(math.pi/2)
            QgAF[i][j] = Tg / (1+math.exp(Kg*(GAF[i][j]-Dg)))
            QaAF[i][j] = Ta / (1+math.exp(Ka*(AAF[i][j]-Da)))
            QAF[i][j] = QgAF[i][j]*QaAF[i][j]

    GBF = np.zeros(SFx.shape)
    ABF = np.zeros(SFx.shape)
    QgBF = np.zeros(SFx.shape)
    QaBF = np.zeros(SFx.shape)
    QBF = np.zeros(SFx.shape)
    for i in range(M):
        for j in range(N):
            if gB[i][j] == gF[i][j]:
                GBF[i][j] = gF[i][j]
            else:
                if gF[i][j]==0:
                    GBF[i][j] = 0
                else:
                    GBF[i][j] = gB[i][j]/gF[i][j]
            ABF[i][j] = 1 - abs(aB[i][j]-aF[i][j])/(math.pi/2)
            QgBF[i][j] = Tg / (1 + math.exp(Kg*(GBF[i][j]-Dg)))
            QaBF[i][j] = Ta / (1 + math.exp(Ka*(ABF[i][j]-Da)))
            QBF[i][j] = QgBF[i][j]*QaBF[i][j]
    # compute the QABF
    deno = np.sum(np.sum(gA+gB))
    nume = np.sum(np.sum(matrix_multi(QAF,gA)+matrix_multi(QBF,gB)))
    output = nume/deno
    return  output



def avgGradient(image):
    # image = Image.open(path).convert('L')
    image = image
    width = image.shape[0]
    width = width - 1
    heigt = image.shape[1]
    heigt = heigt - 1
    tmp = 0.0

    for i in range(width):
        for j in range(heigt):
            dx = float(image[i, j + 1]) - float(image[i, j])
            dy = float(image[i + 1, j]) - float(image[i, j])
            ds = math.sqrt((dx * dx + dy * dy) / 2)
            tmp += ds

    imageAG = tmp / (width * heigt)
    return round(imageAG,3)


def ComEntropy(img1, img2):
    width = img1.shape[0]
    hegith = img1.shape[1]
    tmp = np.zeros((256, 256))
    res = 0
    for i in range(width):
        for j in range(hegith):
            val1 = img1[i][j]
            val2 = img2[i][j]
            tmp[val1][val2] = float(tmp[val1][val2] + 1)
    tmp = tmp / (width * hegith)
    for i in range(256):
        for j in range(256):
            if (tmp[i][j] == 0):
                res = res
            else:
                res = res - tmp[i][j] * (math.log(tmp[i][j] / math.log(2.0)))
    return res

def SD(fusion_image):
    h, w = fusion_image.shape
    m = np.mean(fusion_image)
    # print(m, np.sum(fusion_image))
    s = 0
    for i in range(h):
        for j in range(w):
            s = (fusion_image[i, j] - m)**2 + s
    # sd = (s/(h*w))**0.5
    # sd = (s)**0.5/255
    sd = np.std(fusion_image)
    return sd

def MI (path_A,path_B,path_C,path_D,path_E,path_F,path_G):
    A = path_A
    B = path_B
    C = path_C
    D = path_D
    E = path_E
    F = path_F
    G = path_G
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    D = np.array(D)
    E = np.array(E)
    F = np.array(F)
    G = np.array(G)
    mi1 = shannon_entropy(A) + shannon_entropy(G) - ComEntropy(A, G)
    mi2 = shannon_entropy(B) + shannon_entropy(G) - ComEntropy(B, G)
    mi3 = shannon_entropy(C) + shannon_entropy(G) - ComEntropy(C, G)
    mi4 = shannon_entropy(D) + shannon_entropy(G) - ComEntropy(D, G)
    mi5 = shannon_entropy(E) + shannon_entropy(G) - ComEntropy(E, G)
    mi6 = shannon_entropy(F) + shannon_entropy(G) - ComEntropy(F, G)
    mi = ((mi1+mi2+mi3+mi4+mi5)/5+mi6)/2
    return round(mi,3)


def cv2torch(np_img):
    rgb = np.expand_dims(np_img, 0)
    rgb = np.expand_dims(rgb, 0)
    rgb = np.float32(rgb)
    return torch.from_numpy(rgb)


def torch2cv(t_img):
    return np.squeeze(t_img.numpy())

#测试集
if __name__ == '__main__':
    num = 34
    # methed_list = ['GFF', 'GTF', 'MDLatRR', 'MST_SR', 'MSVD']
    # methed_list = ['NSST_PCNN']
    # methed_list = ['densefuse', "ifcnn", "rfn", 'TRP']
    # methed_list = ['densefuse', "rfn", 'TRP']
    methed_list = ['densenet']
    for methed in methed_list:
        en, ag, sd, sf, mi, ssim, ms_ssim = 0, 0, 0, 0, 0, 0, 0
        error = 0
        for i in range(34):
            print(i, '*'*5)
            fuse = cv2.imread(('/media/jin/b/TG/triplenet/test_images/sp/'+methed+'/sp_34/%d/f.png' % (i)), -1)
            s0 = cv2.imread(('/media/jin/b/TG/triplenet/test_images/sp/'+methed+'/sp_34/%d/0.png' % (i)), -1)
            s1 = cv2.imread(('/media/jin/b/TG/triplenet/test_images/sp/'+methed+'/sp_34/%d/1.png' % (i)), -1)
            s2 = cv2.imread(('/media/jin/b/TG/triplenet/test_images/sp/'+methed+'/sp_34/%d/2.png' % (i)), -1)
            s3 = cv2.imread(('/media/jin/b/TG/triplenet/test_images/sp/'+methed+'/sp_34/%d/3.png' % (i)), -1)
            s4 = cv2.imread(('/media/jin/b/TG/triplenet/test_images/sp/'+methed+'/sp_34/%d/4.png' % (i)), -1)
            d = cv2.imread(('/media/jin/b/TG/triplenet/test_images/sp/'+methed+'/sp_34/%d/d.png' % (i)), -1)
            sd_ = SD(fuse)
            en_ = Entropy(fuse)
            ag_ = avgGradient(fuse)
            mi_ = MI(s0,s1,s2,s3,s4,d,fuse)
            sf_ = spatialF(fuse)
            ssim_ = ((metrics.structural_similarity(s0, fuse)+metrics.structural_similarity(s1, fuse)+metrics.structural_similarity(s2, fuse)+metrics.structural_similarity(s3, fuse)+metrics.structural_similarity(s4, fuse))/5+metrics.structural_similarity(d, fuse))/2
            ms_ssim_ = ((msssim(s0, fuse)+msssim(s1, fuse)+msssim(s2, fuse)+msssim(s3, fuse)+msssim(s4, fuse))/5+msssim(d, fuse))/2

            print('entropy:', en_)
            print('AG:', ag_)
            print('MI:', mi_)
            print('SD:', sd_)
            print('ssim:', ssim_, metrics.structural_similarity(d, fuse))
            print('msssim:', ms_ssim_, msssim(d, fuse))
            print('sf:', sf_)

            sd = sd + sd_
            en = en + en_
            ag = ag + ag_
            mi = mi + mi_
            sf = sf + sf_
            ssim = ssim + ssim_
            if not np.isnan(ms_ssim_):
                ms_ssim = ms_ssim + ms_ssim_
            else:
                error += 1


            # print('cross_entropy:', (cross_entropy(image, image1)+cross_entropy(image, image2))/2)
            # print('qabf:', Qabf(image1,image2,image))

            # print('measure.shannon_entropy', measure.shannon_entropy(image, base=2))
            # i = cv2torch(image)
            # i1 = cv2torch(image1)
            # i2 = cv2torch(image2)
            # print(pytorch_msssim.ssim(i, i1), pytorch_msssim.ssim(i, i2))
            # print(ssim(i, i1, win_size=11, data_range=255), ssim(i, i2, win_size=11, data_range=255))
            # print(float(ms_ssim(i, i1)), ms_ssim(i, i2))
            # img_seq = np.zeros( (image.shape[0], image.shape[1], 1, 2) )
            # img_seq[:, :, 0, 0] = image1
            # img_seq[:, :, 0, 1] = image2
            # ms_ssimc = MS_SSIMc()
            # output_image, score = ms_ssimc(img_seq, i)
            # print(np.mean(score))

            # ms = (float(ms_ssim(i, i1)) + float(ms_ssim(i, i2)))/2
            # en += Entropy(image)
            # # print(Entropy(image))
            # ag += avgGradient(image)
            # # mssim += (msssim(image, image1)+msssim(image, image2))/2
            # mssim += ms
            # cen += (cross_entropy(image, image1)+cross_entropy(image, image2))/2
            # qabf += Qabf(image1,image2,image)
            # qabf_new = QABF(image1,image2,image)
            # print(qabf_new)
            # mi += MI(image1,image2,image)
            # m = image.reshape(-1)
            # m1 = image1.reshape(-1)
            # m2 = image2.reshape(-1)
            # # print((mutual_info_score(m1, m)+mutual_info_score(m2, m))/2)
            # # print(mutual_info_score(m, m))
            # # print(MI(image1,image2,image))
            # ssima += (float(ssim(i, i1)) + float(ssim(i, i2)))/2
        print(methed)
        print('SD:', sd/num)
        print('entropy:', en/num)
        print('AG:', ag/num)
        print('ssim:', ssim/num)
        print('msssim:', ms_ssim/(num-error))
        print('sf:', sf/num)
        print('MI:', mi/num)

