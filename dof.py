import math
import cv2
import argparse
from torch.types import Device
from tqdm import tqdm

import numpy as np
import numpy.ma as ma

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision import models
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

clickFlg = False
mouseX = None
mouseY = None

def clickEvent(event,x,y,flags,param):
	global clickFlg, mouseX, mouseY
	if event == cv2.EVENT_LBUTTONDOWN:
		clickFlg = True
		mouseX, mouseY = x,y
		#print(mouseX, mouseY)


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(
		description='It will blurout your image'
	)

	parser.add_argument(
		'-m', '--model',
		help='Choose model from resnet18, resnet50, densenet121, unet or densenet169',
		default='densenet169',
		type=str
	)

	parser.add_argument(
		'-i', '--input', 
		help='Please provide a path to the input image',
		default=None,
		type=str
	)

	parser.add_argument(
		'-w', '--weight',
		help='Please provide a path to the pretrained weights',
		default=None,
		type=str
	)

	device = None

	if torch.cuda.is_available():
		print("Using the GPU. You are good to go!")
		device = torch.device('cuda:0')
	else:
		device = torch.device('cpu')

	args = parser.parse_args()
	print(args.model)

	img_path = args.input
	original_img = None
	tensor_img = None
	model = None
	original_shape = None
	target_depth = 0.5
	scale = 0.18

	if args.input is not None:
		original_img = cv2.imread(img_path)
		original_shape = original_img.shape
		img = cv2.resize(original_img, (512, 384))
		tensor_img = Image.fromarray(img).convert('RGB')
		tensor_img = ToTensor()(tensor_img).unsqueeze(0)
		print(tensor_img.size())
		

	if args.model == 'resnet18':
		from resnet18 import Dense
	elif args.model == 'resnet50':
		from resnet50 import Dense
	elif args.model == 'densenet121':
		from densenet121 import Dense
	elif args.model == 'unet':
		from unet import Dense
	else:
		from densenet169 import Dense

	
	model = Dense().to(device)
	
	if args.weight is not None:
		input_model = torch.load(args.weight, map_location=device)
		if type(input_model) is dict:	
			model.load_state_dict(input_model['model_state_dict'])
		else:
			model = input_model
		

	with torch.no_grad():
		model = model.eval()
		tensor_img.to(device)
		depthImg = model(tensor_img.float()).cpu().numpy()
		normMap = (depthImg-depthImg.min())
		normMap /= normMap.max()
		print(normMap.shape)
		cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
		cv2.setMouseCallback('image',clickEvent)
		
		aresult = cv2.resize(normMap[0][0,:,:], (int(original_shape[1]*scale), int(original_shape[0]*scale)), interpolation = cv2.INTER_AREA)
		aresult = aresult * 255
		aresult = np.uint8(aresult)
		aresult = cv2.applyColorMap(aresult, cv2.COLORMAP_JET)

		while True:
			if clickFlg == True:
				clickFlg = False
				xx = int(mouseX/(original_shape[1]*scale)*normMap.shape[3])
				yy = int(mouseY/(original_shape[0]*scale)*normMap.shape[2])
				target_depth = normMap[0][0][yy][xx]

			baiasMap = abs(normMap - target_depth)
			baiasMap /= baiasMap.max()
			reDepth = cv2.resize(baiasMap[0][0,:,:], (original_shape[1], original_shape[0]))
			reDepth = np.dstack((reDepth,reDepth,reDepth))
			
			firstLayer= np.where(reDepth <= 0.1, original_img, 0)
			secondLayer = np.where(abs(reDepth - 0.4) < 0.3, cv2.boxFilter(original_img,-1,(31,31)), firstLayer)
			thirdLayer = np.where(reDepth >= 0.7, cv2.boxFilter(original_img,-1,(51,51)), secondLayer)			
			
			rresult = cv2.resize(thirdLayer, (int(original_shape[1]*scale), int(original_shape[0]*scale)), interpolation = cv2.INTER_AREA)
			dresult = cv2.resize(reDepth, (int(original_shape[1]*scale), int(original_shape[0]*scale)), interpolation = cv2.INTER_AREA)
			
			dresult = cv2.applyColorMap(np.uint8(dresult*255), cv2.COLORMAP_JET)
			
			#cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
			
			cv2.imshow('absolute depth', aresult)
			cv2.imshow('relative depth', dresult)
			cv2.imshow('image', rresult)
			if cv2.waitKey(1) == ord('q'):
				break
		
		cv2.destroyAllWindows()
