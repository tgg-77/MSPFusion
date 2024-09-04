# test phase
import torch
from torch.autograd import Variable
import numpy as np
import time
import cv2
import os
import scipy.io as io
import pytorch_msssim
import time



for i in range(17, 18):
    ir_src = cv2.imread('/media/jin/b/TG/triplenet/test_images/TNO/' + 'IR' + '%d.png' % (i + 1))
    vis_src = cv2.imread('/media/jin/b/TG/triplenet/test_images/TNO/' + 'VIS' + '%d.png' % (i + 1))
    h, w, c = ir_src.shape

    x_min = 165
    x_max = 242
    y_min = 253
    y_max = 338
    cv2.rectangle(ir_src, (y_min, x_min), (y_max, x_max), (0, 0, 255), 2)
    cv2.rectangle(vis_src, (y_min, x_min), (y_max, x_max), (0, 0, 255), 2)
    temp_ir = ir_src[x_min+1:x_max, y_min+1:y_max]
    temp_vis = vis_src[x_min+1:x_max, y_min+1:y_max]
    temp_ir = cv2.resize(temp_ir, (0, 0), fx=1.5, fy=1.5)
    temp_vis = cv2.resize(temp_vis, (0, 0), fx=1.5, fy=1.5)
    temp_h, temp_w, _ = temp_ir.shape
    print(temp_h, temp_w)
    # ir_src[h-temp_h:, w-temp_w:, :] = temp_ir
    # vis_src[h-temp_h:, w-temp_w:, :] = temp_vis
    ir_src[h-temp_h:, -temp_w:, :] = temp_ir
    vis_src[h-temp_h:, -temp_w:, :] = temp_vis
    cv2.imwrite('./pt/18/ir%d.png' % i, ir_src)
    cv2.imwrite('./pt/18/vis%d.png' % i, vis_src)


    methed_list = ['SL1_SSIM_96_1e0_gelu', 'SL1_SSIM_96_1e1_gelu', 'SL1_SSIM_96_1e2_gelu', 'SL1_SSIM_96_1e3_gelu', 'SL1_SSIM_96_1e4_gelu', 'ddcgan', 'GANMcC', 'deepfuse', 'densefuse', 'fusiongan', 'ifcnn', 'rfn', 'TRP', 'U2fusion', 'GFF', 'GTF', 'MDLatRR', 'MST_SR', 'MSVD']
    for m in range(len(methed_list)):
        if m > 13:
            image1 = cv2.imread('/media/jin/b/TG/triplenet/test_images/matlab/results/TNO/' + methed_list[m] + '/%s%d.png' % (methed_list[m], i+1))
            ir_src = cv2.imread('/media/jin/b/TG/triplenet/test_images/matlab/images/TNO/' + 'IR' + '%d.png' % (i+1))
            vis_src = cv2.imread('/media/jin/b/TG/triplenet/test_images/matlab/images/TNO/' + 'VIS' + '%d.png' % (i+1))
            h, w, c = ir_src.shape

            x_min = 165
            x_max = 242
            y_min = 253
            y_max = 338
            cv2.rectangle(ir_src, (y_min, x_min), (y_max, x_max), (0, 0, 255), 2)
            cv2.rectangle(vis_src, (y_min, x_min), (y_max, x_max), (0, 0, 255), 2)
            if ir_src.shape != image1.shape:
                image1 = np.resize(image1, ir_src.shape)
        else:
            image1 = cv2.imread('/media/jin/b/TG/triplenet/test_images/TNO/' + methed_list[m] + '/%d.png' % (i + 1))

        cv2.rectangle(image1, (y_min, x_min), (y_max, x_max), (0, 0, 255), 2)
        temp_ir = image1[x_min+1:x_max, y_min+1:y_max]
        temp_ir = cv2.resize(temp_ir, (0, 0), fx=1.5, fy=1.5)
        temp_h, temp_w, _ = temp_ir.shape
        print(temp_h, temp_w)
        # image1[h-temp_h:, -temp_w:, :] = temp_ir
        image1[h-temp_h:, -temp_w:, :] = temp_ir
        cv2.imwrite('./pt/18/%s.png' % methed_list[m], image1)



    # cv2.imwrite('./pt/%d.png' % i, cube)

    #
    # methed = 'TNO'
    # image = cv2.imread('/media/jin/b/TG/triplenet/test_images/matlab/results/TNO/' + methed + '/%s%d.png' % (methed, i + 1), 0)
    #
