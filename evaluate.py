import transform
import torchvision.transforms as tvTransform
import data
import argparse
from torch.backends import cudnn
import torch
from upsample import FCN
import tqdm
import os
import loss
import numpy as np
import matplotlib.pyplot as plt
import timeit
from torch.autograd import Variable
import scipy
from  scipy import ndimage
from piwise.network import FCN8, FCN16, FCN32, UNet, PSPNet, SegNet
import math
import SimpleITK as sitk

def parser_init(parser):
	"""initialize parse arguments"""
	parser.add_argument('--data-folder', type=str, default='./data', metavar='PATH',\
		help='path to data folder')
	parser.add_argument('--no-cuda', action='store_true', default=False, \
		help='disables CUDA training')
	parser.add_argument('--snapshot', type=str, default='./snapshot', metavar='PATH', \
		help='snapshot save location')
	
	args = parser.parse_args()

	args.cuda = not args.no_cuda and torch.cuda.is_available()

	# change parser value here
	args.snapshot = '../../snapshot-unet/snapshot_2500.pth.tar'
	args.data_folder = '../../data/evaluate'
	args.model = 'unet'
	# args.snapshot = 'J:/Deep_Learning/RA_wrist_segmentation/snapshot'
	# args.resume = 'J:/Deep_Learning/RA_wrist_segmentation/snapshot/snapshot_100.pth.tar'
	# args.data_folder = 'J:/Deep_Learning/RA_wrist_segmentation/data'
	# args.cuda = False
	
	return args

def load_data(data_path,workers=0):
	# apply transform to input data, support multithreaded reading, num_workers=0 refers to use main thread
	data_transform = tvTransform.Compose([transform.Normalization(),\
		transform.SitkToTensor()])

	# img_transform = tvTransform.Compose([tvTransform.ToTensor()])

	# load data
	data_set = data.NiftiDataSet(os.path.join(data_path),transform=data_transform,train=False)
	data_loader = torch.utils.data.DataLoader(data_set, batch_size=1,shuffle=False,num_workers=workers,pin_memory=True)

	return data_loader

def evaluate(model,data_loader,cuda=True):
	"""test accuracy of the training model"""
	model.eval()

	for batch_idx, data in enumerate(data_loader):
		image = data['image'][0]

		seg = torch.LongTensor(image.size()).zero_()
		weight = torch.FloatTensor(image.size()).zero_() # eleminate stride overlapping region

		# convert image to 3 channel
		img_tmp = np.zeros((image.size(0),3,image.size(2),image.size(3),image.size(4)))
		img_tmp[:,0,:,:,:] = image.numpy()
		img_tmp[:,1,:,:,:] = image.numpy()
		img_tmp[:,2,:,:,:] = image.numpy()

		# imag data need to normalize to [0,1]
		img_tmp = (img_tmp - np.amin(img_tmp))/(np.amax(img_tmp)-np.amin(img_tmp))

		image = torch.from_numpy(img_tmp).float()

		if cuda:
			# image = image.cuda()
			seg = seg.cuda()
			weight = weight.cuda()

		# count the number to loop over the image, note that img is a 5D tensor
		patch_size = 256
		stride = 256
		inum = int(math.ceil((image.size()[3] - patch_size) / float(stride))) + 1 
		jnum = int(math.ceil((image.size()[4] - patch_size) / float(stride))) + 1
		# knum = int(math.ceil((image.size()[4] - patch_size) / float(stride))) + 1

		patch_tot = 0
		total_count = inum*jnum*image.size()[4]

		count = 0
		for i in range(inum):
			for j in range(jnum):
				for k in range(image.size(2)):
					count = count + 1

					istart = i * stride
					if istart + patch_size > image.size()[3]: #for last patch
						istart = image.size()[3] - patch_size 
					iend = istart + patch_size

					jstart = j * stride
					if jstart + patch_size > image.size()[4]: #for last patch
						jstart = image.size()[4] - patch_size 
					jend = jstart + patch_size

					# kstart = k * stride
					# if kstart + patch_size > img.size()[4]: #for last patch
					# 	kstart = img.size()[4] - patch_size 
					# kend = kstart + patch_size
					patch_tot += 1

					patch_data = image[:,:,k,istart:iend, jstart:jend]

					if cuda:
						patch_data = patch_data.cuda()
					patch_data = Variable(patch_data, volatile=True)

					original_shape = patch_data[0, 0].size()
					output = model(patch_data)

					# # segmentation label output
					output_max = output[0].data.max(0)[1][0]

					# # plot the inference result
					# fig1 = plt.figure(1)
					# # fig1.suptitle('Test Accuracy: {:.6f}'.format(accuracy))
					# plt.ion()
					# plt.subplot(1,2,1)
					# plt.imshow(patch_data.data.cpu().numpy()[0,0,...],cmap='gray')
					# plt.axis('off')

					# plt.subplot(1,2,2)
					# plt.imshow(output_max.cpu().numpy(),cmap='jet')
					# plt.axis('off')
					
					seg[0,0,k,istart:iend, jstart:jend] = \
					torch.add(seg[0,0,k,istart:iend, jstart:jend],output_max)
					weight[:,:,k,istart:iend, jstart:jend] = \
					torch.add(weight[0, 0,k,istart:iend, jstart:jend],1)

	seg = torch.div(seg,weight.long())

	batch_size = 1
	# save result
	for i in range(batch_size):
		# convert back to sitk image
		seg_np = np.squeeze(seg[i,:,:,:,:].cpu().numpy(), axis=0)

		seg_sitk = sitk.GetImageFromArray(seg_np.astype('uint16')) # the label image should use uint16 type

		castFilter = sitk.CastImageFilter()
		castFilter.SetOutputPixelType(1)
		seg_sitk = castFilter.Execute(seg_sitk)

		# not a good practice to save output here, need to refine i/o structure
		reader = sitk.ImageFileReader()
		reader.SetFileName(data_loader.dataset.GetImagePath(batch_idx*batch_size+i))
		img_original = reader.Execute()

		seg_sitk.SetSpacing(img_original.GetSpacing())
		seg_sitk.SetOrigin(img_original.GetOrigin())
		seg_sitk.SetDirection(img_original.GetDirection())
		
		folder = os.path.dirname(data_loader.dataset.GetImagePath(batch_idx*batch_size+i))
		writer = sitk.ImageFileWriter()
		writer.SetFileName(os.path.join(folder,'seg.nii'))
		writer.Execute(seg_sitk)

	# return seg_sitk

def main(args):
	if args.cuda and torch.cuda.is_available():
		print('CUDA acceleration: Yes')
	else:
		if not torch.cuda.is_available():
			print('CUDA device not found')
		print('CUDA acceleration: No')

		# create network model
	classes = 3

	model = None
	if args.model == 'fcn8':
		model = FCN8(classes)
	if args.model == 'fcn16':
		model = FCN16(classes)
	if args.model == 'fcn32':
		model = FCN32(classes)
	if args.model == 'unet':
		model = UNet(classes)
	if args.model == 'pspnet':
		model = PSPNet(classes)
	if args.model == 'segnet':
		model = SegNet(classes)
	if args.model == 'resnet50' or args.model == 'resnet101':
		model = FCN(classes)
	assert model is not None, 'model {args.model} not available'

	if args.cuda and torch.cuda.is_available():
		model = torch.nn.DataParallel(model)
		model.cuda()

	if os.path.isfile(args.snapshot):
		print("Loading snapshot '{}'".format(args.snapshot))
		snapshot = torch.load(args.snapshot)
		model.load_state_dict(snapshot['state_dict'])
		print("=> Snapshot '{}' loaded (epoch {})"
			.format(args.snapshot, snapshot['epoch']))
	else:
		print("No checkpoint found at '{}', evaluation abort".format(args.snapshot))

	# load data
	workers = 1
	print 'loading data...'
	data_loader = load_data(args.data_folder)
	print 'finish loading data'
	evaluate(model,data_loader,args.cuda)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='ResNet Segmenation Model (Training and Testing)')
	args = parser_init(parser)
	main(args)