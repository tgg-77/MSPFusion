# Training DenseFuse network
# auto-encoder

import os
import sys
import time
import numpy as np
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from torch.autograd import Variable
import utils
from MSPFusion_train import MSPFusion
from args_fusion import args
import pytorch_msssim
import torch.nn.functional as F


def main():
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"
	original_imgs_path = utils.list_images(args.dataset)
	train_num = 80000
	original_imgs_path = original_imgs_path[:train_num]
	random.shuffle(original_imgs_path)
	for i in range(1):
		# i = 3
		train(i, original_imgs_path)


def train(i, original_imgs_path):

	batch_size = args.batch_size
	MSPFusion_model = MSPFusion()

	if args.resume is not None:
		print('Resuming, initializing using weight from {}.'.format(args.resume))
		MSPFusion_model.load_state_dict(torch.load(args.resume))
	print(MSPFusion_model)
	optimizer = Adam(MSPFusion_model.parameters(), args.lr)
	L2_loss = torch.nn.MSELoss()
	ssim_loss = pytorch_msssim.msssim

	if args.cuda:
		MSPFusion_model.cuda()

	tbar = trange(args.epochs)
	print('Start training.....')

	# creating save path
	temp_path_model = os.path.join(args.save_model_dir, args.ssim_path[i])
	if os.path.exists(temp_path_model) is False:
		os.makedirs(temp_path_model)

	temp_path_loss = os.path.join(args.save_loss_dir, args.ssim_path[i])
	if os.path.exists(temp_path_loss) is False:
		os.makedirs(temp_path_loss)

	Loss_pixel = []
	Loss_ssim = []
	Loss_l1 = []
	Loss_all = []
	all_ssim_loss = 0.
	all_pixel_loss = 0.
	for e in tbar:
		print('Epoch %d.....' % e)
		# load training database
		image_set_ir, batches = utils.load_dataset(original_imgs_path, batch_size)
		MSPFusion_model.train()
		count = 0
		for batch in range(batches):
			image_paths = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
			img = utils.get_train_images_auto(image_paths, height=args.HEIGHT, width=args.WIDTH)

			count += 1
			optimizer.zero_grad()
			img = Variable(img/255, requires_grad=False)
			# if args.cuda:
			img = img.cuda()
			MSPFusion_model.cuda()
			# print(img.dtype)
			# get fusion image
			# encoder
			en = MSPFusion_model.encoder(img)
			# decoder
			outputs = MSPFusion_model.decoder(en)
			# resolution loss
			x = Variable(img.data.clone(), requires_grad=False)
			outputs = [outputs]
			ssim_loss_value = 0.
			pixel_loss_value = 0.
			L2_loss_value = 0.

			for output in outputs:
				ssim_loss_temp = ssim_loss(output, x, normalize=True)
				l2_loss_temp = L2_loss(output, x)
				ssim_loss_value += (1-ssim_loss_temp)
				L2_loss_value += l2_loss_temp
			ssim_loss_value /= len(outputs)
			L2_loss_value /= len(outputs)
			# total loss
			total_loss = args.ssim_weight[i] * ssim_loss_value + L2_loss_value
			total_loss.backward()
			optimizer.step()

			all_ssim_loss += ssim_loss_value.item()

			all_pixel_loss += L2_loss_value.item()
			if (batch + 1) % args.log_interval == 0:
				mesg = "{}\tEpoch {}:\t[{}/{}]\t pixel loss: {:.6f}\t ssim loss: {:.6f}\t total: {:.6f}".format(
					time.ctime(), e + 1, count, batches,
								  all_pixel_loss / args.log_interval,
								  all_ssim_loss / args.log_interval,
								  (args.ssim_weight[i] * all_ssim_loss + all_pixel_loss) / args.log_interval
				)
				tbar.set_description(mesg)
				Loss_pixel.append(all_pixel_loss / args.log_interval)
				Loss_ssim.append(all_ssim_loss / args.log_interval)
				Loss_all.append((args.ssim_weight[i] * all_ssim_loss + all_pixel_loss) / args.log_interval)

				all_ssim_loss = 0.
				all_pixel_loss = 0.

			if (batch + 1) % (200 * args.log_interval) == 0:
				# save model
				MSPFusion_model.eval()
				MSPFusion_model.cpu()
				save_model_filename = args.ssim_path[i] + '/' + "Epoch_" + str(e) + "_iters_" + str(count) + "_" + \
									  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[
										  i] + ".model"
				save_model_path = os.path.join(args.save_model_dir, save_model_filename)
				torch.save(MSPFusion_model.state_dict(), save_model_path)
				# save loss data
				# pixel loss
				loss_data_pixel = np.array(Loss_pixel)
				loss_filename_path = args.ssim_path[i] + '/' + "loss_pixel_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
				scio.savemat(save_loss_path, {'loss_pixel': loss_data_pixel})
				# SSIM loss
				loss_data_ssim = np.array(Loss_ssim)
				loss_filename_path = args.ssim_path[i] + '/' + "loss_ssim_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
				scio.savemat(save_loss_path, {'loss_ssim': loss_data_ssim})
				# all loss
				loss_data_total = np.array(Loss_all)
				loss_filename_path = args.ssim_path[i] + '/' + "loss_total_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
				scio.savemat(save_loss_path, {'loss_total': loss_data_total})

				MSPFusion_model.train()
				MSPFusion_model.cuda()
				tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)

	# pixel loss
	loss_data_pixel = np.array(Loss_pixel)
	loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_pixel_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':','_') + "_" + \
						 args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_pixel': loss_data_pixel})
	# SSIM loss
	loss_data_ssim = np.array(Loss_ssim)
	loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_ssim_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
						 args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_ssim': loss_data_ssim})
	# all loss
	loss_data_total = np.array(Loss_all)
	loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_total_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
						 args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_total': loss_data_total})
	# save model
	MSPFusion_model.eval()
	MSPFusion_model.cpu()
	save_model_filename = args.ssim_path[i] + '/' "Final_epoch_" + str(args.epochs) + "_" + \
						  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".model"
	save_model_path = os.path.join(args.save_model_dir, save_model_filename)
	torch.save(MSPFusion_model.state_dict(), save_model_path)

	print("\nDone, trained model saved at", save_model_path)


if __name__ == "__main__":
	main()
