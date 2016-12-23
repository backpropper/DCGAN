require 'torch'
require 'optim'
require 'os'
require 'xlua'
require 'nn'

local ffi = require 'ffi'
local lmdb = require 'lmdb'
local tnt = require 'torchnet'
local image = require 'image'
local util = require 'utils'
local model = require 'model'

local optParser = require 'opts'
local opt = optParser.getOpt()

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
	print("Generate Index: ",opt.generateIndex)
	if opt.generateIndex == 'true' then
		class_size = util.generateIndex(path,dataset)
	else
		for i,v in pairs(dataset) do
			hashmap = torch.load(path..v..'_train_lmdb_hashmaps.t7')
			class_size[i] = hashmap:size(1)
		end
	end
else
	local classes = util.getClasses(path)

	for i,v in pairs(classes) do
		local imagepath = path..v
		print("Cropping Images of class: ",imagepath)
		if opt.cropImages then
			util.cropImages(imagepath)
		end
		class_size[i] = util.getdataSize(imagepath)
	end
end

local datasize = 0

for i,s in pairs(class_size) do
	datasize = datasize + s
end



local criterion = nn.BCECriterion()


function getIterator()
	return tnt.ParallelDatasetIterator{
		init    = function() 
			local tnt = require 'torchnet'
			paths.dofile(filename)
			end,
		nthread = opt.nThreads,
		closure = function()
			local image = require 'image'

		    return tnt.BatchDataset{
		    	batchsize = opt.batchSize,
		      		dataset = tnt.ShuffleDataset{
		      			dataset = tnt.ListDataset{
		      				list = torch.range(1, torch.floor(datasize)/400):long(),
		      				load = function(idx)
		      						return {
		      							input =  getSample(idx),
		      							target = getLabel(idx)
		     						}
		   							end
		 				}
					}
				}
				end
	}
end



local input = torch.Tensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
local noise = torch.Tensor(opt.batchSize, opt.inpdim, 1, 1)
local label = torch.Tensor(opt.batchSize)
local disError, genError
local real_label = 1
local fake_label = 0

local gen = {}
local dis = {}

if opt.imageSize == 128 then
	gen = model.generator_128(opt.inpdim, opt.ndisfil, opt.ngenfil, 3)
	dis = model.discriminator_128(opt.inpdim, opt.ndisfil, opt.ngenfil, 3)
elseif opt.imageSize == 64 then
	dis = model.discriminator(opt.inpdim, opt.ndisfil, opt.ngenfil, 3)
	gen = model.generator(opt.inpdim, opt.ndisfil, opt.ngenfil, 3)
else
	dis = model.discriminator_256(opt.inpdim, opt.ndisfil, opt.ngenfil, 3)
	gen = model.generator_256(opt.inpdim, opt.ndisfil, opt.ngenfil, 3)
end


if opt.gpu then
	require 'cunn'
	input = input:cuda()  
	noise = noise:cuda()  
	label = label:cuda()
	require 'cudnn'
	cudnn.benchmark = true
	cudnn.fastest = true
	cudnn.convert(gen, cudnn)
	cudnn.convert(dis, cudnn)
	dis:cuda() 
	gen:cuda()
	criterion:cuda()
end



local disParam, disGradParam = dis:getParameters()
local genParam, genGradParam = gen:getParameters()

if opt.noise == 'uniform' then
	noise:uniform(-1, 1)
	elseif opt.noise == 'normal' then
		noise:normal(0, 1)
	end

	genOptim = {
	learningRate = opt.lr,
	beta1 = opt.beta1,
}
disOptim = {
learningRate = opt.lr,
beta1 = opt.beta1,
}

local disTrain = function(x)

label:fill(real_label) 

dis:forward(input)
local disError_real = criterion:forward(dis.output, label)
disGradParam:zero()
criterion:backward(dis.output, label)
dis:backward(input, criterion.gradInput)

if opt.noise == 'uniform' then
	noise:uniform(-1, 1)
	elseif opt.noise == 'normal' then
		noise:normal(0, 1)
	end
	local fake = gen:forward(noise)
	input:copy(fake)
	label:fill(fake_label)

	dis:forward(input)
	local disError_fake = criterion:forward(dis.output, label)
	criterion:backward(dis.output, label)
	dis:backward(input, criterion.gradInput)

	disError = disError_real + disError_fake
	return disError, disGradParam
end

local genTrain = function(x)
if opt.noise == 'uniform' then
	noise:uniform(-1, 1)
	elseif opt.noise == 'normal' then
		noise:normal(0, 1)
	end
	local fake = gen:forward(noise)
	input:copy(fake)
	label:fill(real_label)

	dis:forward(input)
	genError = criterion:forward(dis.output, label)
	criterion:backward(dis.output, label)
	local df_dg = dis:updateGradInput(input, criterion.gradInput)
	genGradParam:zero()
	gen:backward(noise, df_dg)
	return genError, genGradParam
end

iterator = getIterator()
for epoch = 1, opt.niter do
	local counter = 0
	tempiterator=iterator
	for sample in tempiterator:run() do
		if sample.input:size(1) == opt.batchSize then
			input:copy(sample.input)
			optim.adam(disTrain, disParam, disOptim)
			optim.adam(genTrain, genParam, genOptim)
			counter = counter + 1
			print("Batch No. " .. counter .. " done")
		end
	end
	disParam, disGradParam = dis:getParameters()
	genParam, genGradParam = gen:getParameters()

	torch.save('lsunmodel/'..opt.name .. '_' .. epoch .. '_gen.t7', gen:clearState())
	torch.save('lsunmodel/'..opt.name .. '_' .. epoch .. '_dis.t7', dis:clearState())
	print("Epoch " .. epoch .. " done")
end

torch.save(opt.name .. '_gen.t7', gen:clearState())
torch.save(opt.name ..'_dis.t7', dis:clearState())
