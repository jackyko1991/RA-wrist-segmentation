import SimpleITK as sitk
import torch
import os

class NiftiDataSet(torch.utils.data.Dataset):
	"""use train mode to load image label pair, else only load image. Train mode is also useful in testing phase.
	transform is a general transform apply to both image and label, pririor to img and label transform"""

	def __init__(self, data_folder, transform=None, train=False, img_transform=None, label_transform=None):
		self.data_folder = data_folder
		self.transform = transform
		self.dirlist = os.listdir(data_folder)
		self.train = train
		self.img_transform = img_transform
		self.label_transform = label_transform

	def __checkexist__(self,path):
		if os.path.exists(path):
			return True
		else:
			return False

	def __getitem__(self, index):
		img_name = os.path.join(self.data_folder,self.dirlist[index],'img.nii')

		# check file existence
		if not self.__checkexist__(img_name):
			print(img_name+' not exist!')
			return

		img_reader = sitk.ImageFileReader()
		img_reader.SetFileName(img_name)
		img = img_reader.Execute()

		# add segmentation label to sample if in train mode
		if self.train:
			seg_name = os.path.join(self.data_folder,self.dirlist[index],'label.nii')
			if not self.__checkexist__(seg_name):
				print(seg_name+' not exist!')
				return

			seg_reader = sitk.ImageFileReader()
			seg_reader.SetFileName(seg_name)
			seg = seg_reader.Execute()
		else:
			seg = sitk.Image(img.GetSize(), sitk.sitkInt8)
			seg.SetOrigin(img.GetOrigin())
			seg.SetSpacing(img.GetSpacing())
			seg.CopyInformation(img)

		sample = {'image':img, 'segmentation': seg}

		# apply transform to the data if necessary
		if self.transform:
			sample = self.transform(sample)

		if self.img_transform:
			img = self.img_transform(sample['image'])
		else:
			img = sample['image']
		img = [img]

		if self.label_transform:
			seg = self.label_transform(sample['segmentation'])
		else:
			seg = sample['segmentation']
		seg = [seg]

		sample = {'image':img, 'segmentation': seg}

		return sample 

	def __len__(self):
		return len(self.dirlist)

	def GetImagePath(self,index):
		assert isinstance(index,int)
		assert (index >=0 and index<self.__len__)
		return os.path.join(self.data_folder,self.dirlist[index],'img.nii')

	def GetLabelPath(self,index):
		assert isinstance(index,int)
		assert (index >=0 and index<self.__len__)
		return os.path.join(self.data_folder,self.dirlist[index],'label.nii')