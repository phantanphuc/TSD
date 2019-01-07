import os


obj_list = {}

count = 0

folderlist = ['00']

with open('D:\dataset\main\TSD\System\dataset\BelgiumTSD_annotations\BTSD_training_GTclear.txt', 'r') as file:
	lines = file.readlines()
	for line in lines:
		# for folder in folderlist:
			# if folder in line[:3]:
				# print(line) 
				info = line.replace('\n', '').split(';')

				filename = info[0][3:]
				bounding_box = (info[1], info[2], info[3], info[4], info[5])

				if filename in obj_list:
					obj_list[filename] = obj_list[filename] + [bounding_box]
				else :
					obj_list[filename] = [bounding_box]


				count = count + 1

with open('train.txt', 'w') as fout:
	for record in obj_list:

		write_data = obj_list[record]

		towrite = record + ' ' + str(len(write_data))

		for bb in write_data:
			towrite = towrite + ' ' + bb[0] + ' ' + bb[1] + ' ' + bb[2] + ' ' + bb[3] + ' ' + bb[4]

		fout.write(towrite + '\n')

		print(obj_list[record])
