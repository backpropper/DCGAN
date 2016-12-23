import os
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir("test/gen_images") if isfile(join("test/gen_images", f))]

os.system("python test/labels/makelabels.py test")

totalfiles = len(onlyfiles)
batches = totalfiles / 50

for batch in range(batches):
	cmd = "th test/labels/classify.lua test/labels/resnet-101.t7"
	
	for f in range(50):
		if totalfiles <= (batch*50 + f):
			break
		cmd = cmd + " test/gen_images/" + onlyfiles[batch*50 + f]
	cmd = cmd + " > test/labels/labels_test.txt"
	#print cmd
	os.system(cmd)
	os.system("python test/labels/createlabels.py test")
