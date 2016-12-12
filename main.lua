require 'torch'
require 'optim'
require 'os'
require 'xlua'
require 'nn'

local ffi = require 'ffi'
local lmdb = require 'lmdb'
local tnt = require 'torchnet'
local image = require 'image'
local optParser = require 'opts'
local opt = optParser.parse(arg)
local util = require 'utils'
local model = require 'model'

opt.manualSeed = torch.random(1, 10000)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

local path = '/Users/abhinav/Documents/Acads/CV/Project/myproj/'
local dataset = {'classroom'}
local size
local indexvals

if opt.generateIndex == true then
	datasize = util.generateIndex(path,'classroom')
end

local datasize = {}
local loadData = {}

for i,v in pairs(dataset) do
    loadData[i] = {}
    datasize[i] = {}
    for j=1,3 do
        loadData[i][j]={}
    end
    loadData[i][1] = torch.load(path..v..'_train_lmdb_hashes_chartensor.t7')
	datasize[i] = loadData[i][1]:size(1)
	loadData[i][2] = 	lmdb.env{Path = path..v..'_train_lmdb',
							RDONLY=true, NOLOCK=true, NOTLS=true, NOSYNC=true, NOMETASYNC=true,
							MaxReaders=20, MaxDBs=20}
	loadData[i][2]:open()
	loadData[i][3] = loadData[i][2]:txn(true)
end

local gen = model.generator(opt.inpdim, opt.ndisfil, opt.ngenfil, 3)
local dis = model.discriminator(opt.inpdim, opt.ndisfil, opt.ngenfil, 3)
-- print("Model", gen)
-- print("Model", dis)
local criterion = nn.BCECriterion()

local function loadImage(pic)
   local img = image.decompress(pic, 3, 'float')
   local width = img:size(3)
   local height = img:size(2)

   local outwidth = opt.loadSize
   local outheight = opt.loadSize

   if width < height then
      outheight = outheight * (height/width)
   else
      outwidth = outwidth * (width/height)
   end
   input = image.scale(img, outwidth, outheight)
   return input
end

local preprocess = function(pic)

   local img = loadImage(pic)

   local outWidth = opt.fineSize
   local outHeight = opt.fineSize
   
   local width = img:size(3)
   local height = img:size(2)

   local h = math.ceil(torch.uniform(1e-2, height-outHeight))
   local w = math.ceil(torch.uniform(1e-2, width-outWidth))
   
   local output = image.crop(img, w, h, w + outWidth, h + outHeight)
   local prob = torch.uniform()
   
   if prob > 0.5 then 
      output = image.hflip(output); 
   end
   output:mul(2)
   output:add(-1) 
   return output
end


function getlsunSample(idx)
  local hash = ffi.string(loadData[1][1][idx]:data(), loadData[1][1]:size(2))
  img = loadData[1][3]:get(hash)
  return preprocess(img)
end

function getrealLabel(idx)
    return torch.LongTensor{1}
end

dataset = tnt.ListDataset{
    list = torch.range(1, datasize[1]):long(),
    load = function(idx)
        return {
            input = getlsunSample(idx),
            target = getrealLabel(idx)
        }
    end
}

function getIterator()
    return tnt.DatasetIterator{
        dataset = tnt.BatchDataset{
            batchsize = opt.batchSize,
            dataset = dataset
        }
    }
end

local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local noise = torch.Tensor(opt.batchSize, opt.inpdim, 1, 1)
local label = torch.Tensor(opt.batchSize)
local disError, genError
local real_label = 1
local fake_label = 0

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

for epoch = 1, opt.niter do
   local counter = 0
   for sample in getIterator():run() do
        input:copy(sample.input)
	    optim.adam(disTrain, disParam, disOptim)
		optim.adam(genTrain, genParam, genOptim)
		counter = counter + 1
        print("Batch No. " .. counter .. " done")
	end
	disParam, disGradParam = dis:getParameters()
    genParam, genGradParam = gen:getParameters()
    torch.save(opt.name .. '_' .. epoch .. '_gen.t7', gen:clearState())
    torch.save(opt.name .. '_' .. epoch .. '_dis.t7', dis:clearState())
    print("Epoch " .. epoch .. " done")
end

torch.save(opt.name .. '_gen.t7', gen:clearState())
torch.save(opt.name ..'_dis.t7', dis:clearState())

