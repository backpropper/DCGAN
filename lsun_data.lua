require 'image'
require 'torch'
tds=require 'tds'
require 'lmdb'
ffi = require 'ffi'


local optParser = require 'opts'
local opt = optParser.getOpt(arg)


local path = opt.path
local dataset = opt.dataset

print("Parallel Thread Started")
os.execute('cd ' .. path)

local loadSize   = {3, opt.imageSize * 1.5}
local sampleSize = {3, opt.imageSize}

local function loadImage(img)
 local input = image.decompress(img, 3, 'float')

   local iW = input:size(3)
   local iH = input:size(2)
   if iW < iH then
    input = image.scale(input, loadSize[2], loadSize[2] * iH / iW)
  else
    input = image.scale(input, loadSize[2] * iW / iH, loadSize[2])
  end
  return input
end


local datasize = {}
local cum_size={}
local loadData = {}
local totalsize = 0

cum_size[0] = 0

for i,v in pairs(dataset) do
  loadData[i] = {}
  datasize[i] = {}
  
  for j=1,3 do
    loadData[i][j]={}
  end
  
  loadData[i][1] = torch.load(path..v..'_train_lmdb_hashmaps.t7')
  datasize[i] = loadData[i][1]:size(1)
  totalsize = totalsize + datasize[i]
  cum_size[i] = totalsize

  loadData[i][2] =  lmdb.env{Path = path..v..'_train_lmdb',
                              RDONLY=true, NOLOCK=true, NOTLS=true, NOSYNC=true, NOMETASYNC=true,
                              MaxReaders=20, MaxDBs=20}
  loadData[i][2]:open()
  loadData[i][3] = loadData[i][2]:txn(true)
end




local function loadImage(pic)
  local img = image.decompress(pic, 3, 'float')
  local width = img:size(3)
  local height = img:size(2)

  local outwidth = opt.imageSize*1.5
  local outheight = opt.imageSize*1.5

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
  
  local outWidth = opt.imageSize
  local outHeight = opt.imageSize
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

function getSample(idx)
  classid = 1
  while cum_size[classid] < idx do
    classid = classid+1
  end

  nidx = idx - cum_size[classid-1]

  local hash = ffi.string(loadData[classid][1][nidx]:data(), loadData[classid][1]:size(2))
  img = loadData[1][3]:get(hash)
  return preprocess(img)
end

function getImage(idx)
  classid = 1
  while cum_size[classid] < idx do
    classid = classid+1
  end
  nidx = idx - cum_size[classid-1]
  
  local hash = ffi.string(loadData[classid][1][nidx]:data(), loadData[classid][1]:size(2))
  img = loadData[1][3]:get(hash)
  return image.decompress(img, 3, 'float')
end

function getLabel(idx)
  classid = 1
  while cum_size[classid] < idx do
    classid = classid+1
  end

  nidx = idx - cum_size[classid-1]

  return torch.LongTensor{classid}
end
