import cPickle as pickle
import operator

outfile = open("test/labels/labeldict_train.p", "r")
train = pickle.load(outfile)
outfile.close()
outfile = open("test/labels/labeldict_test.p", "r")
test = pickle.load(outfile)
outfile.close()

toptrain = dict(sorted(train.iteritems(), key=operator.itemgetter(1), reverse=True)[:5])
toptest = dict(sorted(test.iteritems(), key=operator.itemgetter(1), reverse=True)[:5])

print "Real images top-5 labes", toptrain
print "Generated images top-5 labels", toptest