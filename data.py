import SimpleITK as sitk
import torch

class NiftiDataSet(torch.utils.data.Dataset):
	"""use train mode to load image label pair, else only load image. Train mode is also useful in testing phase"""

	def __init__(self, data_folder, transform=None, train=False):
		self.data_folder = data_folder
		self.transform = transform
		self.dirlist = os.listdir(data_folder)
		self.train = train

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