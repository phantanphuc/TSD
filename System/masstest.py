import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.autograd import Variable

from NetworkConfig import *

from ssd import SSD300
from encoder import DataEncoder
from PIL import Image, ImageDraw, ImageFont

import os



DRC = '/home/phucpt2/WORK/LVTN/TSD/System/dataset/train'
output = './testresult'


dictionary = {}
dictindex = []

with open('./label.txt') as f:
	content = f.readlines()
	for symbol in content:
		symbol = symbol.replace('\n','')

		split = symbol.split(' ')

		dictindex.append(split[0])

net = SSD300()
checkpoint = torch.load('./model/model_2019_01_16_E14.pth', map_location='cpu')
checkpoint['net']
net.load_state_dict(checkpoint['net'])
net.eval()


sentinel = 0

for a, b, c in os.walk(DRC):
	for file in c:
		# try:
			# Load test image
			img = Image.open(DRC + '/' + file)
			img1 = img.convert('L').resize((InputImgSize,InputImgSize))
			transform = transforms.Compose([transforms.ToTensor(),
											transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
			img1 = transform(img1)

			# Forward
			loc, conf = net(Variable(img1[None,:,:,:], volatile=True))

			# Decode
			data_encoder = DataEncoder()
			boxes, labels, scores = data_encoder.decode(loc.data.squeeze(0), F.softmax(conf.squeeze(0)).data)

			fnt = ImageFont.truetype('./font/arial.ttf', 40)


			draw = ImageDraw.Draw(img)

			for i in range(len(boxes)):
				boxes[i][::2] *= img.width
				boxes[i][1::2] *= img.height
				draw.rectangle(list(boxes[i]), outline='red')

				print(boxes)
				print(labels.numpy())

				print((boxes[i][0], boxes[i][1]))
				print(labels.numpy()[i, 0] - 2)
				print('------------------------------')

				draw.text((boxes[i][0].item(), boxes[i][1].item()), str(labels.numpy()[i, 0] - 2), font=ImageFont.truetype("./font/arial.ttf"))


			print('saving to: ' + output + '/' + file)
			img.save(output + '/' + file)
			

			sentinel = sentinel + 1
			if sentinel > 21:
				break

		# except:
		# 	print('err')

