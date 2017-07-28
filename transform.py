import SimpleITK as sitk
from PIL import Image
import collections
import numpy as np
import random
import torch

class Normalization(object):
	"""Normalize an image by setting its mean to zero and variance to one."""

	def __call__(self, sample):
		self.normalizeFilter = sitk.NormalizeImageFilter()
		# print("Normalizing image...")
		img, seg = sample['image'], sample['segmentation']
		img = self.normalizeFilter.Execute(img)

		return {'image': img, 'segmentation': seg}

class SitkToTensor(object):
	"""Convert sitk image to Tensors"""

	def __call__(self, sample):
		img, seg = sample['image'], sample['segmentation']

		img_np = sitk.GetArrayFromImage(img)

		img_np = np.float32(img_np)
		img_np_4D = np.zeros((1,img_np.shape[0],img_np.shape[1],img_np.shape[2]))
		img_np_4D[0,:,:,:] = img_np
		img_tensor = torch.from_numpy(img_np_4D).float()

		seg_np = sitk.GetArrayFromImage(seg)
		seg_np = np.uint8(seg_np)
		seg_np_4D = np.zeros((1,seg_np.shape[0],seg_np.shape[1],seg_np.shape[2]))
		seg_np_4D[0,:,:,:] = seg_np
		seg_tensor = torch.from_numpy(seg_np_4D).long()

		return {'image': img_tensor, 'segmentation': seg_tensor}

class SitkToNumpy(object):
	"""Convert sitk image to numpy arrays"""

	def __call__(self,sample):
		img, seg = sample['image'], sample['segmentation']

		img_np = sitk.GetArrayFromImage(img)

		img_np = np.float32(img_np)

		seg_np = sitk.GetArrayFromImage(seg)
		seg_np = np.uint8(seg_np)

		return {'image': img_np, 'segmentation': seg_np}

class NumpyToPIL(object):
	"""Convert sitk image to numpy arrays, make sure that the image value is normalized between 0 and 1"""

	def __call__(self,sample):
		img, seg = sample['image'], sample['segmentation']

		img = np.squeeze(img)
		seg = np.squeeze(seg)

		img = Image.fromarray(np.uint8(img*255))

		# # convert image to 3 channel
		# img3C = np.zeros((img.size[0],img.size[1],3))
		# img3C[:,:,0] = img
		# img3C[:,:,1] = img
		# img3C[:,:,2] = img
		seg = Image.fromarray(seg)

		return {'image': img, 'segmentation': seg}

class Padding(object):
	"""Add padding to the image if size is smaller than patch size

	Args:
		output_size (tuple or int): Desired output size. If int, a cubic volume is formed
	"""

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size, output_size)
		else:
			assert len(output_size) == 3
			self.output_size = output_size

		assert all(i > 0 for i in list(self.output_size))

	def __call__(self,sample):
		img, seg = sample['image'], sample['segmentation']
		size_old = img.GetSize()

		if (size_old[0] >= self.output_size[0]) and (size_old[1] >= self.output_size[1]) and (size_old[2] >= self.output_size[2]):
			return sample
		else:
			resampler = sitk.ResampleImageFilter()
			resampler.SetInterpolator(2)
			resampler.SetOutputSpacing(img.GetSpacing())
			resampler.SetSize(self.output_size)

			# resample on image
			resampler.SetOutputOrigin(img.GetOrigin())
			resampler.SetOutputDirection(img.GetDirection())
			img = resampler.Execute(img)

			# resample on label
			resampler.SetOutputOrigin(seg.GetOrigin())
			resampler.SetOutputDirection(seg.GetDirection())
			seg = resampler.Execute(seg)

			return {'image': img, 'segmentation': seg}

class RandomCrop(object):
	"""Crop randomly the image in a sample. This is usually used for datat augmentation.
		Drop ratio is implemented for randomly dropout crops with empty label. (Default to be 0.2)
		This transformation only applicable in train mode
    Args:
        output_size (tuple or int): Desired output size. If int, cubic crop is made.
    """

	def __init__(self, output_size, drop_ratio=0.1):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size, 1)
		else:
			assert len(output_size) == 2
			self.output_size = (output_size[0],output_size[1],1)

		assert isinstance(drop_ratio, float)
		if drop_ratio >=0 and drop_ratio<=1:
			self.drop_ratio = drop_ratio
		else:
			raise RuntimeError('Drop ratio should be between 0 and 1')

	def __call__(self,sample):
		img, seg = sample['image'], sample['segmentation']
		size_old = img.GetSize()
		size_new = self.output_size

		contain_label = False

		roiFilter = sitk.RegionOfInterestImageFilter()
		roiFilter.SetSize([size_new[0],size_new[1],size_new[2]])

		while not contain_label: 
			# get the start crop coordinate in ijk
			if size_old == size_new:
				[start_i, start_j, start_k] = [0,0,0]
			else:
				start_i = np.random.randint(0, size_old[0]-size_new[0])
				start_j = np.random.randint(0, size_old[1]-size_new[1])
				start_k = np.random.randint(0, size_old[2]-size_new[2])

			roiFilter.SetIndex([start_i,start_j,start_k])

			seg_crop = roiFilter.Execute(seg)
			statFilter = sitk.StatisticsImageFilter()
			statFilter.Execute(seg_crop)

			# will iterate until a sub volume containing label is extracted
			if statFilter.GetSum()<1:
				contain_label = self.drop(self.drop_ratio) # has some probabilty to contain patch with empty label
			else:
				contain_label = True

		img_crop = roiFilter.Execute(img)

		return {'image': img_crop, 'segmentation': seg_crop}

	def drop(self,probability):
		return random.random() <= probability

class ToLabel(object):
	def __call__(self, inputs):
		tensors = []
		for i in inputs:
			tensors.append(torch.from_numpy(np.array(i)).long())
		return tensors

class ReLabel(object):
	def __init__(self, olabel, nlabel):
		self.olabel = olabel
		self.nlabel = nlabel

	def __call__(self, inputs):
		# assert isinstance(input, torch.LongTensor), 'tensor needs to be LongTensor'
		for i in inputs:
			i[i == self.olabel] = self.nlabel
		return inputs

class ToSP(object):
	def __init__(self, size):
		self.scale2 = Scale(int(size/2), Image.NEAREST)
		self.scale4 = Scale(int(size/4), Image.NEAREST)
		self.scale8 = Scale(int(size/8), Image.NEAREST)
		self.scale16 = Scale(int(size/16), Image.NEAREST)
		self.scale32 = Scale(int(size/32), Image.NEAREST)

	def __call__(self, input):
		input2 = self.scale2(input)
		input4 = self.scale4(input)
		input8 = self.scale8(input)
		input16 = self.scale16(input)
		input32 = self.scale32(input)
		inputs = [input, input2, input4, input8, input16, input32]
		# inputs = [input]

		return inputs

class Scale(object):
	def __init__(self, size, interpolation=Image.BILINEAR):
		assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
		self.size = size
		self.interpolation = interpolation

	def __call__(self, img):
		if isinstance(self.size, int):
			w, h = img.size
			if (w <= h and w == self.size) or (h <= w and h == self.size):
				return img
			if w < h:
				ow = self.size
				oh = int(self.size * h / w)
				return img.resize((ow, oh), self.interpolation)
			else:
				oh = self.size
				ow = int(self.size * w / h)
				return img.resize((ow, oh), self.interpolation)
		else:
			return img.resize(self.size, self.interpolation)