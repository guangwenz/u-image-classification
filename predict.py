import os
import time

import logging 
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import torch
from torchvision import models, transforms
import torch.nn.functional as F
import PIL
from PIL import Image

class InferenceNN(object):
	"""docstring for InferenceNN"""
	def __init__(self, checkpoint_path, category_names_path):
		super(InferenceNN, self).__init__()
		self.logger = logging.getLogger("InferenceNN")

		self.checkpoint_path = checkpoint_path
		self.model, self.optimizer, self.class_to_idx, self.classes = self.__load_model(self.checkpoint_path)
		if category_names_path:
			import json
			with open(category_names_path, 'r') as f:
				self.category_names = json.load(f)
		else:
			self.category_names = {}
	def __load_model(self, checkpoint_path):
		start = time.time()

		self.logger.info("loading model from saved checkpoint {}".format(checkpoint_path))

		chk = torch.load(checkpoint_path)
		self.logger.info("data loaded from checkpoint with keys {}".format(chk.keys()))

		model = chk['model']
		model.classifier = chk['classifier']
		model.load_state_dict(chk['state_dict'])
		model.eval()

		end = time.time()
		duration = end - start
		self.logger.info("Model loaded within {:.0f} minutes {:.0f} seconds.".format(duration // 60, duration % 60))

		return (model, chk['optimizer'], chk['class_to_idx'], chk['classes'])
	def __process_image(self, image_path):
		image_transforms = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.485, 0.456, 0.406],
				[0.229, 0.224, 0.225])
			])
		img = Image.open(image_path)
		return image_transforms(img)
	def validate(self, image_path, top_k, gpu):
		device = 'cuda' if gpu and torch.cuda.is_available() else 'cpu'
		self.logger.info("running on {}".format(device))
		model = self.model
		model.to(device)
		with torch.no_grad():
			inputs = self.__process_image(image_path).float().unsqueeze_(0).to(device)
			outputs = model(inputs)
			outputs = F.softmax(outputs,dim=1)
			possibilities, preds = torch.topk(outputs, top_k, dim=1)

			if preds.size(1) == 1:
				top_category = self.classes[preds.data.item()]
				top_category = self.category_names.get(top_category, top_category)
				self.logger.info("The predicated class is === {}({:.3f}) ===".format(top_category, possibilities.data.item()))
			else:
				possibilities = possibilities.squeeze().tolist()
				for ii, (cls_name) in enumerate([self.classes[int(i)] for i in preds.squeeze().tolist()]):
					cls_name = self.category_names.get(cls_name, cls_name)
					self.logger.info("Top #{}({:.3f}): {}".format(ii+1, possibilities[ii], cls_name))
		
def main():
	import argparse

	parser = argparse.ArgumentParser(description='Inference image class based on checkpoint data')
	parser.add_argument('checkpoint_path', nargs=1, help="checkout datat rebuild the model")
	parser.add_argument('image_path', nargs=1, help="image path as the input")
	parser.add_argument('--category_names', help='Use a mapping of categories to real names')
	parser.add_argument('--top_k', type=int, default=3, help='Use a mapping of categories to real names')
	parser.add_argument('--gpu', action='store_true', help='Set hyperparameters gpu')

	args = parser.parse_args()
	InferenceNN(args.checkpoint_path[0], args.category_names).validate(args.image_path[0], args.top_k, args.gpu)

if __name__ == '__main__':
	main()
