import os
import time
import copy
import logging 
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models, datasets, transforms

from workspace_utils import active_session
class DataProvider(object):
	"""docstring for DataProvider"""
	def __init__(self, data_directory, batch_size=32):
		super(DataProvider, self).__init__()
		self.logger = logging.getLogger("DataProvider")

		self.data_directory = data_directory
		data_transforms = self.__transforms()
		self.logger.info("training data set is loaded from {}".format(data_directory))
		phases = ['train','valid']
		self._image_dataset = {x:datasets.ImageFolder(os.path.join(data_directory, x), data_transforms[x]) for x in phases}
		self._data_loaders = {x:torch.utils.data.DataLoader(self._image_dataset[x], batch_size=batch_size, shuffle=True) for x in phases}
		self._data_sizes = {x: len(self._image_dataset[x]) for x in phases}

	def __transforms(self):
		'''
		build transform for data augmentation
		'''
		train = transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.485, 0.456, 0.406],
				[0.229, 0.224, 0.225])
			])
		valid = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.485, 0.456, 0.406],
				[0.229, 0.224, 0.225])
			])
		return {
			'train': train,
			'valid': valid
		}
	def data_loader(self, phase):
		return self._data_loaders[phase]
	def data_size(self, phase):
		return self._data_sizes[phase]
	def classes(self):
		return self._image_dataset['train'].classes
	def class_to_idx(self):
		return self._image_dataset['train'].class_to_idx
	def out_features(self):
		return len(self._image_dataset['train'].classes)
class NeuralNetwork(object):
	"""docstring for NeuralNetwork"""
	def __init__(self):
		super(NeuralNetwork, self).__init__()

		self.logger = logging.getLogger('NeuralNetwork')
		self.logger.setLevel(logging.DEBUG)

	def __in_features(self, arch):
		if arch == 'vgg13':
			return 25088
		elif arch == 'densenet121':
			return 1024
		else:
			self.logger.error("unknown network arch {}".format(arch))
	def __classifier(self, hidden_units, in_features, out_features):
		classifier = None
		if hidden_units and type(hidden_units) is list:
			classifier = nn.Sequential()
			classifier.add_module('fc0', nn.Linear(in_features, hidden_units[0]))
			classifier.add_module('relu0', nn.ReLU())
			classifier.add_module('dropout0', nn.Dropout(0.5))
			for ii,(hin, hout) in enumerate(zip(hidden_units[:-1], hidden_units[1:])):
				classifier.add_module('fc{}'.format(ii+1), nn.Linear(hin, hout))
				classifier.add_module('relu{}'.format(ii+1), nn.ReLU())
				classifier.add_module('dropout{}'.format(ii+1), nn.Dropout(0.5))
			classifier.add_module('output', nn.Linear(hidden_units[-1], out_features))
		else:
			classifier = nn.Sequential()
			classifier.add_module('fc0', nn.Linear(in_features, 4096))
			classifier.add_module('relu0', nn.ReLU())
			classifier.add_module('dropout0', nn.Dropout(0.5))
			classifier.add_module('fc1', nn.Linear(4096, 4096))
			classifier.add_module('relu1', nn.ReLU())
			classifier.add_module('dropout1', nn.Dropout(0.5))
			classifier.add_module('output', nn.Linear(4096, out_features))

		return classifier

	def __model(self, arch):
		if arch == 'vgg13':
			return models.vgg13(pretrained=True)
		elif arch == 'densenet121':
			return models.densenet121(pretrained=True)
		else:
			self.logger.error("unknown arch {}".format(arch))
			return None

	def __build_model(self, classifier, arch):
		model = self.__model(arch)
		if model:
			for param in model.parameters():
				param.require_grad=False
			model.classifier = classifier
			self.logger.info(model)
		return model

	def __do_train(self, model, data_provider, lr, epochs, gpu, momentum=0.9):
		if torch.cuda.is_available():
			self.logger.info("GPU is available! to enable it, pass --gpu")
		if gpu and torch.cuda.is_available():
			device = 'cuda'
		else:
			self.logger.warning('running on cpu, slowness expected!')
			device = 'cpu'
		self.logger.info("training on device {}".format(device))
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(model.classifier.parameters(), lr=lr, momentum=momentum)
		

		best_model_weights = copy.deepcopy(model.state_dict())
		best_accuracy = 0.0

		since = time.time()

		model.to(device)
		for e in range(epochs):
			self.logger.info("Epoch {}/{}".format(e+1, epochs))
			for phase in ['train','valid']:
				is_train = phase == 'train'

				if is_train: 
					model.train()
				else:
					model.eval()

				running_loss = 0
				running_corrects = 0				
				for inputs,labels in data_provider.data_loader(phase):
					inputs, labels = inputs.to(device), labels.to(device)

					optimizer.zero_grad()

					with torch.set_grad_enabled(is_train):
						outputs = model(inputs)
						_, preds = torch.max(outputs, 1)
						loss = criterion(outputs, labels)
						if is_train:
							loss.backward()
							optimizer.step()
					running_loss += loss.item() * inputs.size(0)
					running_corrects += torch.sum(preds == labels)
				epoch_loss = running_loss / data_provider.data_size(phase)
				epochs_accuracy = running_corrects.double() / data_provider.data_size(phase)
				self.logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epochs_accuracy))

				if epochs_accuracy > best_accuracy and not is_train:
					best_accuracy = epochs_accuracy
					best_model_weights = copy.deepcopy(model.state_dict())

		duration = time.time() - since
		self.logger.info("Training completed within {:.0f} minutes {:.0f} seconds".format(duration // 60, duration % 60))
		self.logger.info('Best val Accuracy: {:4f}'.format(best_accuracy))
		model.load_state_dict(best_model_weights)
		return (model, optimizer)

	def __save_checkpoint(self, data_provider, save_dir, model, classifier, optimizer, epochs, arch, lr, hidden_units, gpu):
		data = {
		"in_features": self.__in_features(arch),
		"out_features": data_provider.out_features(),
		"epochs": epochs,
		"arch": arch,
		"lr":lr,
		"hidden_units":hidden_units,
		"gpu":gpu,
		'class_to_idx': data_provider.class_to_idx(),
		'classes': data_provider.classes(),

		"classifier":classifier,
		"model": self.__model(arch),
		"optimizer": optimizer.state_dict(),
		'state_dict': model.state_dict()
		}
		save_dir = save_dir if save_dir else os.path.dirname(os.path.realpath(__file__))
		target = os.path.join(save_dir, 'train_checkpoint_{}.pth'.format(time.strftime('%Y%m%d_%H%M%S')))
		self.logger.info("saving checkpoint data to {} for data with {} items".format(target, len(data)))
		torch.save(data, target)
	def train(self, data_directory, save_dir, arch, lr, hidden_units, epochs, gpu):
		'''
		Train a neural network with specificed arch and hyper parameters
		'''
		self.logger.debug('Training neural network using arch {} with hyperparameters lr: {}, hidden_units:{}, epochs:{}, GPU: {}'.format(arch, lr, hidden_units, epochs, gpu))
		
		data_provider = DataProvider(data_directory)
		classifier = self.__classifier(hidden_units, self.__in_features(arch), data_provider.out_features())
		model = self.__build_model(classifier, arch)
		with active_session():
			model, optimizer = self.__do_train(model, data_provider, lr, epochs, gpu=gpu)
		self.__save_checkpoint(data_provider, save_dir, model, classifier, optimizer, epochs, arch, lr, hidden_units, gpu)		
		
def main():
	import argparse

	parser = argparse.ArgumentParser(description='Train a new network on a data set')
	parser.add_argument('data_directory', nargs=1, help='data directory holding training and validation data set')
	parser.add_argument('--save_dir', dest='save_dir', help='Set directory to save checkpoints, default current directory')
	parser.add_argument('--arch', dest='arch', default='vgg13', choices=['vgg13','densenet121'], help='Choose architecture, default is vgg13')
	parser.add_argument('--learning_rate', dest='learning_rate', default=0.001, type=float, help='Set hyperparameters learning rate, default is 0.001')
	parser.add_argument('--hidden_units', dest='hidden_units', help='Set hyperparameters hidden units, could be a single number or numbers separated with ,')
	parser.add_argument('--epochs', dest='epochs', default=10, type=int, help='Set hyperparameters epochs, default is 10')
	parser.add_argument('--gpu', dest='gpu', action='store_true', help='Set hyperparameters gpu, default false')

	args = parser.parse_args()
	print(args)
	NeuralNetwork().train(args.data_directory[0], args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)
if __name__ == '__main__':
	main()