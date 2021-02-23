import torchvision.models as models 
import torch.nn as nn 

class PretrainedNets:
			
	def getresnet18():
		resnet18 = models.resnet18(pretrained=True)
		# Freeze model weights
		for param in resnet18.parameters():
			param.requires_grad = False
		num_ftrs = resnet18.fc.in_features
		# Here the size of each output sample is set to 2.
		# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
		resnet18.fc = nn.Linear(num_ftrs, 2)
		resnet18.name = 'resnet18'
		return resnet18
		
	def getdensenet161():
		densenet161 = models.densenet161(pretrained=True)
		for param in densenet161.parameters():
			param.requires_grad = False
		num_ftrs = densenet161.classifier.in_features
		densenet161.classifier = nn.Linear(num_ftrs, 2)
		densenet161.name = 'densenet161'
		return densenet161
		
	def getvgg16():
		vgg16 = models.vgg16(pretrained=True)
		# Freeze model weights
		for param in vgg16.parameters():
			param.requires_grad = False
		# Add on classifier
		vgg16.classifier[6] = nn.Sequential(
							  nn.Linear(4096, 256), 
							  nn.ReLU(), 
							  nn.Dropout(0.4),
							  nn.Linear(256, 2),                   
							  nn.LogSoftmax(dim=1))
		vgg16.name = 'vgg16'
		return vgg16
		
	def getpretrainednet(self,netName):
		if netName == "resnet18":
			return getresnet18()
		elif netName == "densenet161":
			return getdensenet161()
		elif netName == "vgg16":
			return getvgg16()
		else:
			print('Invalid pretrained net! Enter following available net\n resnet18\n densenet161\n vgg16\n')
			return -1