require 'image'
require 'torch'
tds=require 'tds'
require 'lmdb'
ffi = require 'ffi'


local optParser = require 'opts'
local opt = optParser.getOpt(arg)
local util = require 'utils'


local path = opt.path
local classes = util.getClasses(path)

local loadData = {}
local class_size = {}
local cum_size = {}
local totalsize = 0

cum_size[0] = 0

for i,v in pairs(classes) do
  print("Classes: ",v)
  loadData[i] = {}
  local imagepath = path..v
  print("Imagepath: ",imagepath)
  imagetable = util.getImages(imagepath)
  loadData[i] = imagetable
  class_size[i] = #loadData[i]
  totalsize = totalsize+class_size[i]
  cum_size[i] = totalsize
end

--------------------------------------------------------------


local loadSize   = {3, opt.imageSize*1.5}
local sampleSize = {3, opt.imageSize}



local function loadImage(imgpath)
  local img = image.load(imgpath)
  local width = img:size(3)
  local height = img:size(2)

  local outwidth = opt.imageSize*1.5
  local outheight = opt.imageSize*1.5

  input = image.scale(img, outwidth, outheight)
  return input
end



local preprocess = function(imgpath)
  --print("Preprocess Start") 
  local img = loadImage(imgpath)
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
  --print("Preprocess End") 
  return output
end

function getImage(idx)
  classid = 1
  while cum_size[classid] < idx do
    classid = classid+1
  end
  nidx = idx - cum_size[classid-1]
  imagename = loadData[classid][nidx]
  imgpath = path..classes[classid]..'/'..imagename
  return image.load(imgpath)
end

function getSample(idx)
  classid = 1
  while cum_size[classid] < idx do
    classid = classid+1
  end
  nidx = idx - cum_size[classid-1]
  imagename = loadData[classid][nidx]
  imgpath = path..classes[classid]..'/'..imagename
  return preprocess(imgpath)
end

function getLabel(idx)
  classid = 1
  while cum_size[classid] < idx do
    classid = classid+1
  end

  nidx = idx - cum_size[classid-1]

  return torch.LongTensor{classid}
end