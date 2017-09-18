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
from piwise.criterion import CrossEntropyLoss2d

def parser_init(parser):
	"""initialize parse arguments"""
	parser.add_argument('--data-folder', type=str, default='./data', metavar='PATH',\
		help='path to data folder')
	parser.add_argument('--train-batch-size', type=int, default=16, metavar='N',\
		help='input batch size for training (default: 16)')
	parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',\
		help='input batch size for testing (default: 4)')
	parser.add_argument('--patch-size', type=int, default=64, metavar='N',\
		help='input small patch size')
	parser.add_argument('--epochs', type=int, default=50, metavar='N',\
		help='number of epochs to train (default: 50)')
	parser.add_argument('--lr', type=float, default=0.01, metavar='LR', \
		help='learning rate (default: 0.01)')
	parser.add_argument('--momentum', type=float, default=0.5, metavar='M', \
		help='SGD momentum (default: 0.5)')
	parser.add_argument('--decay-weight', type=float, default=1e-8, metavar='N', \
		help='decay weight (default: 1e-8)')
	parser.add_argument('--drop-ratio', type=float, default=0.1, metavar='N', \
		help='random crop drop empty patch probability, drop all empty patch for 0 and accept all empty patch for 1 (default: 0.1)')
	parser.add_argument('--no-cuda', action='store_true', default=False, \
		help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S', \
		help='random seed (default: 1)')
	parser.add_argument('--log-interval', type=int, default=25, metavar='N', \
		help='how many batches to wait before logging training status')
	parser.add_argument('--snapshot', type=str, default='./snapshot', metavar='PATH', \
		help='snapshot save location')
	parser.add_argument('--resume', type=str, metavar='PATH', default='./snapshot/snapshot.pth.tar', \
		help='path to latest snapshot (default: none)')
	parser.add_argument('--model', type=str, metavar='N', default='unet',\
		help='choose network model (fcn8, fcn16, fcn32, [unet], pspnet, segment, resnet50, resnet101)')
	
	args = parser.parse_args()

	args.cuda = not args.no_cuda and torch.cuda.is_available()

	# change parser value here
	args.epochs = 2500
	args.patch_size = 256
	args.train_batch_size = 64
	args.test_batch_size = 1
	args.lr = 1e-4 
	args.decay_weight = 2e-5
	args.momentum = 0.9
	args.model = 'unet'
	args.snapshot = '../../snapshot-unet'
	args.resume = '../../snapshot-unet/snapshot_2500.pth.tar'
	# args.snapshot = '../../snapshot-50'
	# args.resume = '../../snapshot-50/snapshot_1200.pth.tar'
	args.data_folder = '../../data'
	# args.snapshot = 'J:/Deep_Learning/RA_wrist_segmentation/snapshot'
	# args.resume = 'J:/Deep_Learning/RA_wrist_segmentation/snapshot/snapshot_100.pth.tar'
	# args.data_folder = 'J:/Deep_Learning/RA_wrist_segmentation/data'
	args.log_interval = 50
	args.drop_ratio = 0.005
	# args.cuda = False
	
	return args

def load_data(data_path,train_batch_size,test_batch_size,patch_size,workers=0,drop_ratio=0):
	# apply transform to input data, support multithreaded reading, num_workers=0 refers to use main thread

	data_transform = tvTransform.Compose([transform.RandomCrop(patch_size,drop_ratio), \
		transform.Normalization(),\
		transform.SitkToNumpy(),\
		transform.NumpyToPIL(),\
		transform.Rotate()])

	# img_transform = tvTransform.Compose([tvTransform.ToTensor(),\
		# tvTransform.Normalize([.485, .456, .406], [.229, .224, .225])])
	img_transform = tvTransform.Compose([tvTransform.ToTensor()])

	# seg_transform = tvTransform.Compose([transform.ToSP(256), \
	# 	transform.ToLabel(), \
	# 	transform.ReLabel(255, 3)])
	# seg_transform = tvTransform.Compose([transform.ToSP(256), \
	# 	transform.ToLabel(),\
	# 	transform.ReLabel(2, 1)])

	seg_transform = tvTransform.Compose([transform.ToSP(256), \
		transform.ToLabel(),\
		transform.Threshold(0,2)])

	# load data
	train_set = data.NiftiDataSet(os.path.join(data_path,'train'),transform=data_transform,img_transform=img_transform,label_transform=seg_transform,train=True)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size,shuffle=True,num_workers=workers,pin_memory=True)

	test_set = data.NiftiDataSet(os.path.join(data_path,'test'),transform=data_transform,img_transform=img_transform,label_transform=seg_transform,train=True)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size,shuffle=True,num_workers=workers,pin_memory=True)

	return [train_loader,test_loader]

def dice(im1,im2):
	im1 = np.asarray(im1).astype(np.bool)
	im2 = np.asarray(im2).astype(np.bool)

	if im1.shape != im2.shape:
		raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

	# Compute Dice coefficient
	intersection = np.logical_and(im1, im2)

	if im1.sum() + im2.sum() == 0:
		return 1

	return 2. * intersection.sum() / (im1.sum() + im2.sum())

def train(train_loader,epoch,model,optimizer,criterion,cuda=True):
	print('Start epoch {}'.format(epoch))
	"""train the model"""
	model.train()

	epoch_loss = 0

	for batch_idx, data in enumerate(train_loader):
		image = data['image'][0]
		label= data['segmentation'][0][0] # the label is list of mulitresolutional image, 1st data is non-zoomed data
		
		label = label[:,np.newaxis,...]

		# convert image to 3 channel
		img_tmp = np.zeros((image.size(0),3,image.size(2),image.size(3)))
		img_tmp[:,0,:,:] = image.numpy().squeeze()
		img_tmp[:,1,:,:] = image.numpy().squeeze()
		img_tmp[:,2,:,:] = image.numpy().squeeze()

		image = torch.from_numpy(img_tmp).float()

		if cuda and torch.cuda.is_available():
			image = Variable(image.cuda())
			label = Variable(label.cuda())
		else:
			image = Variable(image)
			label = Variable(label)

		# if torch.max(label.data) >2:
		# 	plt.ion()
		# 	plt.subplot(1,2,1)
		# 	plt.imshow(image.data.cpu().numpy()[0,0,:,:],cmap='gray')
		# 	plt.axis('off')
		# 	plt.subplot(1,2,2)
		# 	plt.imshow(label.data.cpu().numpy()[0,0,:,:])
		# 	plt.axis('off')
		# 	plt.draw()
		# 	plt.pause(999999)

		output = model(image)

		optimizer.zero_grad()
		loss = criterion(output,label[:, 0])
		loss.backward()
		optimizer.step()

		epoch_loss += loss.data[0]

	epoch_loss = epoch_loss/batch_idx
	print("Training of epoch {} finished. Average Training Loss: {:.6f}".format(epoch, epoch_loss))

	snapshot = {'epoch': epoch, \
		'state_dict': model.state_dict(), \
		'optimizer': optimizer.state_dict(), \
		'loss': epoch_loss/(batch_idx+1), \
		'epoch_end': True}

	return [epoch_loss, snapshot]

def test(test_loader,epoch,model,cuda=True):
	"""test accuracy of the training model"""
	model.eval()

	test_accuracy = 0

	for batch_idx, data in enumerate(test_loader):
		image = data['image'][0]
		label= data['segmentation'][0][0] # the label is list of mulitresolutional image, 1st data is non-zoomed data
		
		label = label[:,np.newaxis,...]

		# convert image to 3 channel
		img_tmp = np.zeros((image.size(0),3,image.size(2),image.size(3)))
		img_tmp[:,0,:,:] = image.numpy()
		img_tmp[:,1,:,:] = image.numpy()
		img_tmp[:,2,:,:] = image.numpy()

		image = torch.from_numpy(img_tmp).float()

		if cuda and torch.cuda.is_available():
			image = Variable(image.cuda())
			label = Variable(label.cuda())
		else:
			image = Variable(image)
			label = Variable(label)

		output = model(image)

		accuracy = dice(label[0].data.cpu().numpy()[0,...],output[0].data.max(0)[1].cpu().numpy()[0,...])

		# # plot the inference result
		# fig1 = plt.figure(1)
		# fig1.suptitle('Test Accuracy: {:.6f}'.format(accuracy))
		# plt.ion()
		# plt.subplot(1,3,1)
		# plt.imshow(image[0].data.cpu().numpy()[0,...],cmap='gray')
		# plt.axis('off')

		# plt.subplot(1,3,2)
		# plt.axis('off')
		# plt.imshow(label[0].data.cpu().numpy()[0,:,:],cmap='jet')

		# plt.subplot(1,3,3)
		# plt.imshow(output[0].data.max(0)[1].cpu().numpy()[0,...],cmap='jet')
		# plt.axis('off')
		# plt.draw()
		# plt.pause(0.00000001)
		
		test_accuracy = test_accuracy + accuracy

	# compute average accuracy
	test_accuracy = test_accuracy/(batch_idx+1)

	print('Testing of epoch {} finished. Average Testing Accuracy: {:.6f}'.format(
		epoch, test_accuracy))

	return test_accuracy


def train_resnet(train_loader,epoch,model,optimizer,criterion,weight,cuda=True):
	print('Start epoch {}'.format(epoch))
	"""train the model"""
	model.train()

	epoch_loss = 0

	for batch_idx, data in enumerate(train_loader):
	# for batch_idx, (images, labels_group) in tqdm.tqdm(enumerate(train_loader)):
		images = data['image']
		labels_group = data['segmentation']

		if cuda and torch.cuda.is_available():
			images = [Variable(image.cuda()) for image in images]
			labels_group = [labels for labels in labels_group]
		else:
			images = [Variable(image) for image in images]
			labels_group = [labels for labels in labels_group]

		optimizer.zero_grad()
		losses = []
		for img, labels in zip(images, labels_group):
			outputs = model(img)
			
			if cuda and torch.cuda.is_available():
				labels = [Variable(label.cuda()) for label in labels]
			else:
				labels = [Variable(label) for label in labels]
			for pair in zip(outputs, labels):
				losses.append(criterion(pair[0], pair[1]))

			# plt.ion()
			# plt.subplot(1,3,1)
			# plt.imshow(images[0].data.cpu().numpy()[0, 0,:,:],cmap='gray')
			# plt.subplot(1,3,2)
			# plt.imshow(labels[0].data.cpu().numpy()[0,:,:],cmap='jet')
			# plt.subplot(1,3,3)
			# plt.imshow(outputs[0].data.max(0)[1].cpu().numpy()[0,0,:,:],cmap='jet')

			# plt.draw()
			# plt.pause(0.001)

			# exit()

		loss_weight = weight

		loss = 0
		for w, l in zip(loss_weight, losses):
			loss += w*l

		loss.backward()

		optimizer.step()
		epoch_loss += loss.data[0]

		# lr = lr * (1-(92*epoch+i)/max_iters)**0.9
		# for parameters in optimizer.param_groups:
		#     parameters['lr'] = lr
	 
	epoch_loss = epoch_loss/batch_idx
	print("Training of epoch {} finished. Average Training Loss: {:.6f}".format(epoch, epoch_loss))

	snapshot = {'epoch': epoch, \
			'state_dict': model.state_dict(), \
			'optimizer': optimizer.state_dict(), \
			'loss': epoch_loss/(batch_idx+1), \
			'epoch_end': True}

	return [epoch_loss, snapshot]

def test_resnet(test_loader,epoch,model,weight,cuda=True):
	"""test accuracy of the training model"""
	model.eval()

	test_accuracy = 0

	for batch_idx, data in enumerate(test_loader):
		# get the inputs
		images = data['image']
		labels_group = data['segmentation']

		if cuda and torch.cuda.is_available():
			images = [Variable(image.cuda()) for image in images]
			labels_group = [labels for labels in labels_group]
		else:
			images = [Variable(image) for image in images]
			labels_group = [labels for labels in labels_group]

		for img, labels in zip(images, labels_group):
			outputs = model(img)
			
			if cuda and torch.cuda.is_available():
				labels = [Variable(label.cuda()) for label in labels]
			else:
				labels = [Variable(label) for label in labels]

		accuracy = 0
		for i in range(6):
			accuracy = accuracy + dice(labels[i].data.cpu().numpy()[0,:,:],outputs[i].data.max(0)[1].cpu().numpy()[0,0,:,:])*weight[i]

		# plot the inference result
		if accuracy > 0.0:
			fig1 = plt.figure(1)
			fig1.suptitle('Test Accuracy: {:.6f}'.format(accuracy))
			plt.ion()
			plt.subplot(1,3,1)
			plt.imshow(images[0].data.cpu().numpy()[0, 0,:,:],cmap='gray')
			plt.axis('off')

			plt.subplot(1,3,2)
			plt.axis('off')
			plt.imshow(labels[0].data.cpu().numpy()[0,:,:],cmap='jet')

			output_np = np.zeros((outputs[0].data.max(1)[1].cpu().numpy().shape[2],outputs[0].data.max(0)[1].cpu().numpy().shape[3]))
			for i in range(6):
				output_tmp = outputs[i].data.max(0)[1].cpu().numpy()[0,0,:,:]
				output_np = scipy.ndimage.interpolation.zoom(output_tmp,pow(2,i))*weight[i] + output_np

			plt.subplot(1,3,3)
			output_np = np.round(output_np)

			# plt.imshow(outputs[0].data.max(1)[1].cpu().numpy()[0,0,:,:],cmap='jet')
			plt.imshow(output_np,cmap='jet')
			plt.axis('off')

			plt.draw()
			plt.pause(0.00000001)
		
		test_accuracy = test_accuracy + accuracy

	# compute average accuracy
	test_accuracy = test_accuracy/(batch_idx+1)

	print('Testing of epoch {} finished. Average Testing Accuracy: {:.6f}'.format(
		epoch, test_accuracy))

	return test_accuracy

def main(args):
	if args.cuda and torch.cuda.is_available():
		print('CUDA acceleration: Yes')
	else:
		if not torch.cuda.is_available():
			print('CUDA device not found')
		print('CUDA acceleration: No')

	# manual random seed
	torch.manual_seed(args.seed)
	if args.cuda and torch.cuda.is_available():
		torch.cuda.manual_seed(args.seed)

	# cudnn benchmark
	if torch.cuda.is_available():
		cudnn.benchmark = True
	# 	cudnn.benchmark = False # don't use CUDNN if GPU memeory is insufficient

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

	weight = torch.ones(classes)
	# weight = torch.ones(22)
	# weight[0] = 0
	# weight[classes-1] = 0
	# weight[classes-2] = 0

	# weight = torch.ones(22)
	# weight[21] = 0
	# max_iters = 92*epoches

	#load data
	workers = 1
	[train_loader, test_loader] = load_data(args.data_folder, args.train_batch_size, args.test_batch_size, args.patch_size, workers, 0.05)

	if args.cuda and torch.cuda.is_available():
		weight = weight.cuda()

	if args.model.startswith == 'resnet':
		criterion = loss.CrossEntropyLoss2d(weight)
	else:
		criterion = CrossEntropyLoss2d(weight)

	if args.model.startswith('FCN') or args.model.startswith('PSP') or args.model.startswith('resnet'):
		optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay_weight)
	if args.model.startswith('Seg'):
		optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
	else:
		optimizer = torch.optim.Adam(model.parameters())

	# create snapshot folder
	if not os.path.isdir(args.snapshot):
		os.mkdir(args.snapshot)

	start_epoch = 1

	if args.resume:
		if os.path.isfile(args.resume):
			print("Loading snapshot '{}'".format(args.resume))
			snapshot = torch.load(args.resume)
			start_epoch = snapshot['epoch']
			model.load_state_dict(snapshot['state_dict'])
			print("=> Snapshot '{}' loaded (epoch {})"
				.format(args.resume, snapshot['epoch']))
			if snapshot['epoch_end']:
				start_epoch = start_epoch + 1
				# print('Snapshot reach batch end, start next epoch')
		else:
			print("No checkpoint found at '{}', training starts from epoch 1".format(args.resume))

	if args.epochs < start_epoch:
		print("Epoch to train is less than the one in snapshot, training abort")
		return

	# initialize values for plotting
	train_loss_record = np.zeros(args.epochs)
	train_loss_record[:] = np.NAN
	test_accuracy_record = np.empty(args.epochs)
	test_accuracy_record[:] = np.NAN

	if os.path.isfile(args.resume):
		train_loss_record[0:start_epoch-1] = snapshot['train_loss'][0:start_epoch-1]
		test_accuracy_record[0:start_epoch-1] = snapshot['test_accuracy'][0:start_epoch-1]

	# plot loss and accuracy
	fig0 = plt.figure(0)
	plt.ion()
	ax1 = fig0.add_subplot(1, 1, 1)
	# fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.set_xlabel('Epoch')
	ax1.set_ylabel('Loss')
	ax2.set_ylabel('Accuracy')
	ax1.set_title('Epoch: 0, Train Loss: 0, Test Accuracy: 0, Benchmark: 0 s/epoch')
	ax1.set_xlim([1,start_epoch])
	ax1.set_ylim([0,2.5])
	ax2.set_ylim([0,1])
	line1, = ax1.plot(range(1,args.epochs+1), train_loss_record, 'k-',label='Train Loss')
	line4, = ax2.plot(range(1,args.epochs+1), test_accuracy_record, 'b-', label='Test Accuracy')

	#legend
	handles, labels = ax1.get_legend_handles_labels()
	plt.legend([line1, line4], \
		['Train Loss', 'Test Accuracy'],loc=2)
	plt.draw()

	# timer for benchmarking
	timer = timeit.default_timer()
	epoch_count = 1

	for epoch in range(1, args.epochs + 1):
		if epoch >= start_epoch:
			if args.model == 'resnet50' or args.model == 'resnet101':
				# outputs is a multiresolution weighted
				if epoch < 50:
					weight_res = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]
				if epoch < 100 and epoch >= 50:
					weight_res = [0.1, 0.1, 0.1, 0.1, 0.5, 0.1]
				if epoch < 150 and epoch >= 100:
					weight_res = [0.1, 0.1, 0.1, 0.5, 0.1, 0.1]
				if epoch < 200 and epoch >= 150:
					weight_res = [0.1, 0.1, 0.5, 0.1, 0.1, 0.1]
				if epoch < 250 and epoch >= 200:
					weight_res = [0.1, 0.5, 0.1, 0.1, 0.1, 0.1]
				if epoch < 300 and epoch >= 250:
					weight_res = [0.2, 0.4, 0.1, 0.1, 0.1, 0.1]
				if epoch < 350 and epoch >= 300:
					weight_res = [0.3, 0.3, 0.1, 0.1, 0.1, 0.1]
				if epoch < 400 and epoch >= 350:
					weight_res = [0.4, 0.2, 0.1, 0.1, 0.1, 0.1]
				else:
					weight_res = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]

				[epoch_train_loss, snapshot] = train_resnet(train_loader,epoch,model,optimizer,criterion,weight_res,args.cuda)
				epoch_test_accuracy = test_resnet(test_loader,epoch,model,weight_res,args.cuda) # test accuracy after when each epoch train ends
			else:
				[epoch_train_loss, snapshot] = train(train_loader,epoch,model,optimizer,criterion,args.cuda)
				epoch_test_accuracy = test(test_loader,epoch,model,args.cuda) # test accuracy after when each epoch train ends

			# epoch_train_loss = 0
			train_loss_record[epoch-1] = epoch_train_loss

			# epoch_test_accuracy = 1
			test_accuracy_record[epoch-1] = epoch_test_accuracy

			# update train loss plot
			line1.set_ydata(train_loss_record)
			line4.set_ydata(test_accuracy_record)

			# if epoch == start_epoch:
			# 	continue
			# else:
			# 	ax1.set_xlim([start_epoch,epoch])

			ax1.set_xlim([1,epoch])
			# ax1.set_ylim([0,2.5])
			ax1.set_ylim([0,max(train_loss_record)]) # cannot function well when resume training, going to be fixed
			# ax2.set_ylim([0,max(train_accuracy_record)])
			ax1.set_title('Epoch: %s \nTrain Loss: %s, Test Accuracy: %s\n Benchmark: %s s/epoch'\
				%(epoch, \
				"{0:.2f}".format(epoch_train_loss),\
				"{0:.2f}".format(epoch_test_accuracy),\
				"{0:.2f}".format((timeit.default_timer() - timer)/epoch_count)))
			plt.draw()
			plt.pause(0.000000001)

			epoch_count = epoch_count+1

			# save snapshot
			if epoch % args.log_interval == 0:
				snapshot['train_loss'] = train_loss_record
				snapshot['test_accuracy'] = test_accuracy_record
				snapshot_path = args.snapshot + '/snapshot_' + str(epoch) + '.pth.tar'
				torch.save(snapshot, snapshot_path)
				print('Snapshot of epoch {} saved at {}.'.format(epoch, snapshot_path))

	plt.show()


if __name__=="__main__":
	parser = argparse.ArgumentParser(description='ResNet Segmenation Model (Training and Testing)')
	args = parser_init(parser)
	main(args)