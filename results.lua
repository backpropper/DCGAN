require 'torch'
require 'optim'
require 'os'

local ffi = require 'ffi'
local lmdb = require 'lmdb'
local tnt = require 'torchnet'
local image = require 'image'
local util = require 'utils'

local optParser = require 'opts'
local opt = optParser.getOpt()
opt.batchSize = 50
manualSeed = torch.random(1, 10000)
torch.manualSeed(manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

local path = opt.path
local dataset = opt.dataset
local filename = 'lsun_data.lua'

if opt.datatype == 'celeb' then
	filename = 'celeb_data.lua'
end

print("Data Type: ",opt.datatype)
print("Data Path: ",path)

if opt.datatype == 'celeb' then
	dataset = 'img_align_celeba'
end
print("Dataset: ",dataset)

local class_size = {}

if opt.datatype == 'lsun' then
	for i,v in pairs(dataset) do
		hashmap = torch.load(path..v..'_train_lmdb_hashmaps.t7')
		class_size[i] = hashmap:size(1)
	end
else
	local classes = util.getClasses(path)

	for i,v in pairs(classes) do
		local imagepath = path..v
		class_size[i] = util.getdataSize(imagepath)
	end
end

local datasize = 0

for i,s in pairs(class_size) do
	datasize = datasize + s
end

function getIterator()
	return tnt.ParallelDatasetIterator{
		init = function() 
			local tnt = require 'torchnet'
			paths.dofile(filename)
		end,
		nthread = opt.nThreads,
		closure = function()
    
	    return tnt.BatchDataset{
	    	batchsize = opt.batchSize,
	      		dataset = tnt.ShuffleDataset{
	      			dataset = tnt.ListDataset{
	      				list = torch.range(1, torch.floor(datasize)):long(),
	      				load = function(idx)
      						return {
      							input =  getImage(idx)
     						}
   							end
	 				}
			}
		}
		end
	}
end

iterator = getIterator()
local val = 500
os.execute("python test/labels/makelabels.py train")
local imagepaths = {}
for sam in iterator:run() do
	for i = 1,sam.input:size(1) do
		image.save('test/labels/trainimages/'..i..'.jpg',sam.input[i])
		imagepaths[i] = 'test/labels/trainimages/'..i..'.jpg'
	end 
	val = val-1
	imgString = imagepaths[1]
	for i = 2,opt.batchSize do
		imgString = imgString .. " " .. imagepaths[i]
	end
	-- print(imgString)
	labcmd = "th test/labels/classify.lua test/labels/resnet-101.t7 " .. imgString .. " > test/labels/labels_train.txt"
	os.execute(labcmd)
	os.execute("python test/labels/createlabels.py train")

	if val == 0 then
		break
	end
end

