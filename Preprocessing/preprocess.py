import os, re


def preprocess_training_file(path, x_train_path, y_train_path):
	
	first_line = True
	second_line = True

	x_train = []
	y_train = []

	with open(path) as fileobject:
		for line in fileobject:
			data = line.split("\t")
			current_user = data[1]
			text = data[3][:-1].lower()
			text = re.sub('(?<=[a-z])([!?,.])', r' \1', text)

			
			if first_line:
				init_user = current_user
				previous_user = current_user
				x_train.append(text + ' _EOS_ ')
				first_line = False

			elif current_user != init_user and second_line:
				y_train.append(text + ' _EOS_ ')
				second_line = False

			elif init_user == current_user:
				if previous_user == current_user:
					prev_utterance = x_train.pop()
					x_train.append(prev_utterance + " " + text + ' _EOS_ ')
				else:
					x_train.append(text + ' _EOS_ ')
			else:
				if previous_user == current_user:
					prev_utterance = y_train.pop()
					y_train.append(prev_utterance + " " + text + ' _EOS_ ')
				else:
					y_train.append(text + ' _EOS_ ')

			previous_user = current_user

	x_train_file = open(x_train_path, 'a')
	y_train_file = open(y_train_path, 'a')
	print "****"
	print x_train
	print y_train

	for i in range(len(y_train)):

		x_train_file.write(x_train[i] + '_EOT_\n')
		y_train_file.write(y_train[i] + '_EOT_\n')
	x_train_file.close()
	y_train_file.close()

for folder in os.listdir("../../../ubuntu-ranking-dataset-creator/src/dialogs"):
	if folder != ".DS_Store":
		folder_path = "../../../ubuntu-ranking-dataset-creator/src/dialogs/" + folder
		for filename in os.listdir(folder_path):
			file_path = folder_path + "/" + filename
			preprocess_training_file(file_path, "./x_train.txt", "./y_train.txt")





