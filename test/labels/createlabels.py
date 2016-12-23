import cPickle as pickle
import sys

outfile = open("test/labels/labeldict_" + sys.argv[1] + ".p", "r")
classes = pickle.load(outfile)
outfile.close()

with open("test/labels/labels_" + sys.argv[1] + ".txt") as f:
	for i,line in enumerate(f.readlines()):
		data = line.split(None, 1)
		try:
			float(data[0])
		except (IndexError, ValueError):
			continue

		classes[data[1].strip()] += 1
		# print data[1].strip(),classes[data[1].strip()]

outfile = open("test/labels/labeldict_" + sys.argv[1] + ".p", "w")
pickle.dump(classes, outfile)
outfile.close()
