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
	
	args = parser.parse_args()

	args.cuda = not args.no_cuda and torch.cuda.is_available()

	# change parser value here
	args.epochs = 80
	args.patch_size = 256
	args.train_batch_size = 1
	args.test_batch_size = 1
	args.lr = 1e-4 
	args.decay_weight = 2e-5
	args.momentum = 0.9
	args.snapshot = '/mnt/disk0/projects/RA_wrist_segmentation/snapshot'
	args.resume = '/mnt/disk0/projects/RA_wrist_segmentation/snapshot/snapshot_100.pth.tar'
	args.data_folder = '/mnt/disk0/projects/RA_wrist_segmentation/data'
	# args.snapshot = 'J:/Deep_Learning/RA_wrist_segmentation/snapshot'
	# args.resume = 'J:/Deep_Learning/RA_wrist_segmentation/snapshot/snapshot_100.pth.tar'
	# args.data_folder = 'J:/Deep_Learning/RA_wrist_segmentation/data'
	args.drop_ratio = 0.005
	args.cuda = False
	
	return args

def load_data(data_path,train_batch_size,test_batch_size,patch_size,workers=0,drop_ratio=0.1):
	# apply transform to input data, support multithreaded reading, num_workers=0 refers to use main thread

	data_transform = tvTransform.Compose([transform.RandomCrop(patch_size,drop_ratio), \
		transform.Normalization(),\
		transform.SitkToNumpy()])

	img_transform = tvTransform.Compose([tvTransform.ToTensor(),\
		tvTransform.Normalize([.485, .456, .406], [.229, .224, .225])])

	seg_transform = tvTransform.Compose([transform.ToSP(256), \
		transform.ToLabel(), \
		transform.ReLabel(255, 3)])

	# load data
	train_set = data.NiftiDataSet(os.path.join(data_path,'train'),transform=data_transform,img_transform=img_transform,label_transform=seg_transform,train=True)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size,shuffle=True,num_workers=workers,pin_memory=True)

	test_set = data.NiftiDataSet(os.path.join(data_path,'test'),transform=data_transform,img_transform=img_transform,label_transform=seg_transform,train=True)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size,shuffle=True,num_workers=workers,pin_memory=True)

	return [train_loader,test_loader]

def train(train_loader,epoch,model,optimizer,cuda=True):
	print('Start epoch {}'.format(epoch))
	"""train the model"""
	model.train()

	epoch_loss = 0
	epoch_accuracy = 0

	tmp_loss = 0
	tmp_accuracy = 0

	for batch_idx, (images, labels_group) in tqdm.tqdm(enumerate(train_loader)):
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

			print('img',img.size())
			print('labels',labels[0].size())
			print('outputs',outputs[0].size())

			labels = [Variable(label.cuda()) for label in labels]
			for pair in zip(outputs, labels):
				losses.append(criterion(pair[0], pair[1]))

		if epoch < 40:
			loss_weight = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]
		else:
			loss_weight = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]

		loss = 0
		for w, l in zip(loss_weight, losses):
			loss += w*l

		loss.backward()
		optimizer.step()
		running_loss += loss.data[0]

        # lr = lr * (1-(92*epoch+i)/max_iters)**0.9
        # for parameters in optimizer.param_groups:
        #     parameters['lr'] = lr

	print("Epoch [%d] Loss: %.4f" % (epoch+1, running_loss/i))
    # ploter.plot("loss", "train", epoch+1, running_loss/i)
	running_loss = 0

	if (epoch+1) % 20 == 0:
		lr /= 10
		optimizer = torch.optim.SGD(model.parameters(), lr=lr,
			momentum=momentum,
			weight_decay=weight_decay)
        torch.save(model.state_dict(), "./pth/fcn-deconv-%d.pth" % (epoch))



	# for batch_idx, data in enumerate(train_loader):
	# 	if batch_idx % math.ceil(len(train_loader)/10.0) == 0:
	# 		print('Batch {}/{} ({:.2f}%)'.format(batch_idx+1,len(train_loader),100.*(batch_idx+1)/len(train_loader)))

	# 	# get the inputs
	# 	img = data['image']
	# 	label = data['segmentation']
		
	# 	if cuda and torch.cuda.is_available():
	# 		img, label = img.cuda(), label.cuda()

	# 	# wrap them in Variable
	# 	img, label = Variable(img), Variable(label) # convert tensor into variable
		
	# 	optimizer.zero_grad()
	# 	output = model(img)
	# 	label = label.view(label.numel())
	# 	loss = F.nll_loss(output, label)
	# 	loss.backward()
	# 	optimizer.step()

	# 	# compute average loss
	# 	tmp_loss = tmp_loss + loss.data[0]
	# 	epoch_loss = epoch_loss + loss.data[0]

	# 	# compute average accuracy
	# 	pred = output.data.max(1)[1]  # get the index of the max log-probability
	# 	incorrect = pred.ne(label.data).sum()

	# 	tmp_accuracy  = tmp_accuracy + 1.0 - float(incorrect) / label.numel()
	# 	epoch_accuracy = epoch_accuracy + 1.0 - float(incorrect) / label.numel()



		# if (batch_idx+1) == len(train_loader):
		# 	print('Training of epoch {} finished. Average Training Loss/Accuracy: {:.6f}/{:.6f}'.format(
		# 		epoch, epoch_loss/(batch_idx+1), epoch_accuracy/(batch_idx+1)))
		# 	snapshot = {'epoch': epoch, \
		# 	'state_dict': model.state_dict(), \
		# 	'optimizer': optimizer.state_dict(), \
		# 	'loss': epoch_loss/(batch_idx+1), \
		# 	'accuracy': epoch_accuracy/(batch_idx+1), \
		# 	'epoch_end': True}
		# 	# torch.save(snapshot, snapshot_folder + '/snapshot_' + str(epoch) + '_' + str(batch_idx+1))
		# 	return [epoch_loss/(batch_idx+1), epoch_accuracy/(batch_idx+1), snapshot]


def main(args):
	if args.cuda and torch.cuda.is_available():
		print('CUDA acceleration: Yes')
	else:
		print('CUDA acceleration: No')

	# manual random seed
	torch.manual_seed(args.seed)
	if args.cuda and torch.cuda.is_available():
		torch.cuda.manual_seed(args.seed)

	# cudnn benchmark
	if torch.cuda.is_available():
		cudnn.benchmark = True

	# create network model
	model = FCN(22)
	if args.cuda and torch.cuda.is_available():
		model = torch.nn.DataParallel(model)
		model.cuda()

	weight = torch.ones(22)
	weight[21] = 0
	# max_iters = 92*epoches

	#load data
	workers = 30
	[train_loader, test_loader] = load_data(args.data_folder, args.train_batch_size, args.test_batch_size, args.patch_size, workers, 0.05)

	if args.cuda and torch.cuda.is_available():
		weight = weight.cuda()
	criterion = loss.CrossEntropyLoss2d(weight)
	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.decay_weight)

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
	train_accuracy_record = np.empty(args.epochs)
	train_loss_record[:] = np.NAN
	train_accuracy_record[:] = np.NAN
	test_loss_record = np.zeros(args.epochs)
	test_accuracy_record = np.empty(args.epochs)
	test_loss_record[:] = np.NAN
	test_accuracy_record[:] = np.NAN

	if os.path.isfile(args.resume):
		train_loss_record = snapshot['train_loss'] 
		train_accuracy_record = snapshot['train_accuracy']
		test_loss_record = snapshot['test_loss']
		test_accuracy = snapshot['test_accuracy'] 

	# plot loss and accuracy
	plt.ion()

	fig,ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.set_xlabel('Epoch')
	ax1.set_ylabel('Loss')
	ax2.set_ylabel('Accuracy')
	ax1.set_title('Epoch: 0, Train Loss: 0, Test Accuracy: 0, Benchmark: 0 s/epoch')
	ax1.set_xlim([1,2])
	ax1.set_ylim([0,1])
	ax2.set_ylim([0,1])
	line1, = ax1.plot(range(1,args.epochs+1), train_loss_record, 'k-',label='Train Loss')
	line2, = ax2.plot(range(1,args.epochs+1), train_accuracy_record, 'r-',label='Train Accuracy')
	line3, = ax1.plot(range(1,args.epochs+1), test_loss_record, 'g-',label='Test Loss')
	line4, = ax2.plot(range(1,args.epochs+1), test_accuracy_record, 'b-', label='Test Accuracy')

	#legend
	handles, labels = ax1.get_legend_handles_labels()
	plt.legend([line1, line2, line3, line4], \
		['Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy'],loc=2)
	plt.draw()

	# timer for benchmarking
	timer = timeit.default_timer()
	epoch_count = 1

	for epoch in range(1, args.epochs + 1):
		if epoch >= start_epoch:

			[epoch_train_loss, epoch_train_accuracy, snapshot] = train(train_loader,epoch,model,optimizer,args.cuda)
			# [epoch_test_loss,epoch_test_accuracy] = test(test_loader,epoch,model,args.cuda) # test accuracy after when each epoch train ends

			# epoch_train_loss = 0
			# epoch_train_accuracy = 1
			train_loss_record[epoch-1] = epoch_train_loss
			train_accuracy_record[epoch-1] = epoch_train_accuracy

			epoch_test_loss = 0
			epoch_test_accuracy = 1
			test_loss_record[epoch-1] = epoch_test_loss
			test_accuracy_record[epoch-1] = epoch_test_accuracy

			# update train loss plot
 			line1.set_ydata(train_loss_record)
 			line2.set_ydata(train_accuracy_record)
 			line3.set_ydata(test_loss_record)
 			line4.set_ydata(test_accuracy_record)

 			if epoch == start_epoch:
 				continue
 			else:
 				ax1.set_xlim([start_epoch,epoch])
 			ax1.set_ylim([0,1.5])
 			# ax1.set_ylim([0,max(train_loss_record)]) # cannot function well when resume training, going to be fixed
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
				snapshot['train_accuracy'] = train_accuracy_record
				snapshot['test_loss'] = test_loss_record
				snapshot['test_accuracy'] = test_accuracy_record
				snapshot_path = args.snapshot + '/snapshot_' + str(epoch) + '.pth.tar'
				torch.save(snapshot, snapshot_path)
				print('Snapshot of epoch {} saved at {}.'.format(epoch, snapshot_path))

	plt.show()


if __name__=="__main__":
	parser = argparse.ArgumentParser(description='ResNet Segmenation Model (Training and Testing)')
	args = parser_init(parser)
	main(args)