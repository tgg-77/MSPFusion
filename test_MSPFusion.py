# test phase
import torch
from torch.autograd import Variable
from net import *
import utils
from args_fusion import args
import numpy as np
import time
import cv2
import os
import scipy.io as io
import pytorch_msssim
from evaluation_sp import *
import time
import os
from torch.utils.data import Dataset
import random
import cv2
import h5py
from torch.utils.data import DataLoader
import scipy.io as sio
import matplotlib.pyplot as plt


def load_model(path):
    print(path)
    model = model_generator(method='MSPFusion')
    # nest_model = DenseFuse_net(input_nc, output_nc)
    nest_model = model
    nest_model.load_state_dict(torch.load(path))

    para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))

    nest_model.eval()
    nest_model.cuda()

    return nest_model


def _generate_fusion_image(model, h1, h2, h3, h4, h5, dolp):
    # encoder
    w_list = model.fusion_layer(h1, h2, h3, h4, h5, dolp)
    en_1 = model.encoder(h1)
    en_2 = model.encoder(h2)
    en_3 = model.encoder(h3)
    en_4 = model.encoder(h4)
    en_5 = model.encoder(h5)
    en_d = model.encoder(dolp)
    # fusion
    f = model.fusion(en_1, en_2, en_3, en_4, en_5, en_d, w_list)
    # decoder
    img_fusion = model.decoder(f)
    return [img_fusion]



def vision_features(feature_maps, img_type):
    count = 0
    for features in feature_maps:
        count += 1
        for index in range(features.size(1)):
            file_name = 'feature_maps_' + img_type + '_level_' + str(count) + '_channel_' + str(index) + '.png'
            output_path = 'outputs/feature_maps/' + file_name
            map = features[:, index, :, :].view(1, 1, features.size(2), features.size(3))
            map = map * 255
            # save images
            utils.save_image_test(map, output_path)

def img_pr(img_nol):
    img_nol = img_nol.cpu().data[0].numpy()
    img_nol = img_nol.transpose(1, 2, 0)
    # img_nol = (img_nol - np.min(img_nol)) / (np.max(img_nol) - np.min(img_nol))
    # img_nol = img_nol * 255
    return img_nol



def main():

    # run demo
    test_path = "dataset"
    network_type = 'densefuse'
    fusion_type = 'auto'  # auto, fusion_layer, fusion_all
    strategy_type_list = ['addition',
                          'attention_weight']  # addition, attention_weight, attention_enhance, adain_fusion, channel_fusion, saliency_mask
    en, ag, sd, sf, mi, ssim, ms_ssim = 0, 0, 0, 0, 0, 0, 0
    error = 0
    num = 34
    for q in range(1):
        output_path = './output/%d/' % q
        # output_path = 'pol/'
        strategy_type = strategy_type_list[0]
        hyper_path = 'NWPUSP/%d.mat' % q
        # hyper_path = '/media/jin/b/dataset/NWPUSPI/mat_v2/%d.mat' % q
        rgb = sio.loadmat(hyper_path)['input']
        hsi = sio.loadmat(hyper_path)['label']

        if os.path.exists(output_path) is False:
            os.makedirs(output_path)

        model_path = './model/models_trans_l1_nobias_noresize.model'

        h = np.zeros((4, 696, 697, 32), dtype=np.float32)
        r = np.zeros((4, 696, 697, 3), dtype=np.float32)
        for i in range(4):
            h[i, :, :, :] = hsi[:, :, i * 32:(i + 1) * 32]
            r[i, :, :, :] = hsi[:, :, i * 3:(i + 1) * 3]

        with torch.no_grad():
            print('NUMBER ----- ' + str(q))
            # print('SSIM weight ----- ' + args.ssim_path[2])
            ssim_weight_str = args.ssim_path[2]
            model = load_model(model_path)
            out_path = 'NWPUSP'


            if not os.path.exists(out_path):
                os.makedirs(out_path)

            time_list = []
            for i in range(1):
                index = i + 1

                h_raw = h
                h = h.astype(np.float32)
                h = np.sum(h, axis=0) / 2
                h = h[:, :, [5, 10, 15, 20, 25]]
                h = (h - h.min()) / (h.max() - h.min())
                h = torch.tensor(h*255)
                if args.cuda:
                    h = h.cuda()

                h1 = h[:, :, 0]
                h2 = h[:, :, 1]
                h3 = h[:, :, 2]
                h4 = h[:, :, 3]
                h5 = h[:, :, 4]
                r0 = np.mean(h_raw[0, :, :, :], axis=2)*255
                r1 = np.mean(h_raw[1, :, :, :], axis=2)*255
                r2 = np.mean(h_raw[2, :, :, :], axis=2)*255
                r3 = np.mean(h_raw[3, :, :, :], axis=2)*255

                cv2.imwrite(output_path + 'd0.png', r0)
                cv2.imwrite(output_path + 'd1.png', r1)
                cv2.imwrite(output_path + 'd2.png', r2)
                cv2.imwrite(output_path + 'd3.png', r3)

                s0 = (r0 + r1 + r2 + r3) / 2
                s1 = r0 - r2
                s2 = r1 - r3
                dolp = np.sqrt(s1 * s1 + s2 * s2) / (s0)
                dolp[np.isnan(dolp)] = 0



                dolp = ((dolp - dolp.min()) / (dolp.max() - dolp.min()))
                dolp = torch.tensor(dolp*255)

                if args.cuda:
                    dolp = dolp.cuda()
                h1 = h1.unsqueeze(0).unsqueeze(0)
                h2 = h2.unsqueeze(0).unsqueeze(0)
                h3 = h3.unsqueeze(0).unsqueeze(0)
                h4 = h4.unsqueeze(0).unsqueeze(0)
                h5 = h5.unsqueeze(0).unsqueeze(0)
                dolp = dolp.unsqueeze(0).unsqueeze(0)
                h1 = Variable(h1, requires_grad=False)
                h2 = Variable(h2, requires_grad=False)
                h3 = Variable(h3, requires_grad=False)
                h4 = Variable(h4, requires_grad=False)
                h5 = Variable(h5, requires_grad=False)
                dolp = Variable(dolp, requires_grad=False)


                output_path_root = output_path
                temp_time = time.time()
                img_fusion = _generate_fusion_image(model, h1, h2, h3, h4, h5, dolp)
                temp_time_gap = time.time() - temp_time
                time_list.append(temp_time_gap)
                # img_fusion = _generate_fusion_image(model, strategy_type, ir_img, vis_img)
                ############################ multi outputs ##############################################
                file_name = 'fusion_' + fusion_type + '_' + str(
                    index) + '_network_' + network_type + '_' + strategy_type + '_' + ssim_weight_str + '.png'


                img_nol = img_fusion[0].cpu().data[0].numpy()

                img_nol = img_nol.transpose(1, 2, 0)
                # print(np.max(img_nol))
                img_nol = (img_nol-np.min(img_nol))/(np.max(img_nol)-np.min(img_nol))
                img_nol = img_nol * 255
                h1 = img_pr(h1)
                h2 = img_pr(h2)
                h3 = img_pr(h3)
                h4 = img_pr(h4)
                h5 = img_pr(h5)
                dolp = img_pr(dolp)
                if args.cuda:
                    img = img_fusion[0].cpu().clamp(0, 255).data[0].numpy()
                else:
                    img = img_fusion[0].clamp(0, 255).data[0].numpy()

                cv2.imwrite(output_path_root + 'f.png', img_nol)
                cv2.imwrite(output_path_root + '0.png', h1)
                cv2.imwrite(output_path_root + '1.png', h2)
                cv2.imwrite(output_path_root + '2.png', h3)
                cv2.imwrite(output_path_root + '3.png', h4)
                cv2.imwrite(output_path_root + '4.png', h5)
                cv2.imwrite(output_path_root + 'd.png', dolp)

                h1 = h1.squeeze().astype(np.uint8)
                h2 = h2.squeeze().astype(np.uint8)
                h3 = h3.squeeze().astype(np.uint8)
                h4 = h4.squeeze().astype(np.uint8)
                h5 = h5.squeeze().astype(np.uint8)
                dolp = dolp.squeeze().astype(np.uint8)
                img_nol = img_nol.squeeze().astype(np.uint8)


                sd_ = SD(img_nol)
                en_ = Entropy(img_nol)
                ag_ = avgGradient(img_nol)
                mi_ = MI(h1, h2, h3, h4, h5, dolp, img_nol)
                sf_ = spatialF(img_nol)
                ssim_ = ((metrics.structural_similarity(h5, img_nol) + metrics.structural_similarity(h1,
                                                                                                  img_nol) + metrics.structural_similarity(
                    h2, img_nol) + metrics.structural_similarity(h3, img_nol) + metrics.structural_similarity(h4,
                                                                                                        img_nol)) / 5 + metrics.structural_similarity(
                    dolp, img_nol)) / 2
                ms_ssim_ = ((msssim(h5, img_nol) + msssim(h1, img_nol) + msssim(h2, img_nol) + msssim(h3, img_nol) + msssim(h4,
                                                                                                                img_nol)) / 5 + msssim(
                    dolp, img_nol)) / 2
                print('entropy:', en_)
                print('AG:', ag_)
                print('MI:', mi_)
                print('SD:', sd_)
                print('ssim:', ssim_, metrics.structural_similarity(dolp, img_nol))
                print('msssim:', ms_ssim_, msssim(dolp, img_nol))
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

        print('Done......')
    print('SD:', sd / num)
    print('entropy:', en / num)
    print('AG:', ag / num)
    print('ssim:', ssim / num)
    print('sf:', sf / num)
    print('msssim:', ms_ssim / (num-error))
    print('MI:', mi / num)

if __name__ == '__main__':

    main()
