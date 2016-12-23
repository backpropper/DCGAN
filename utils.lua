require 'lmdb'
require 'image'
require 'torch'
tds=require 'tds'
ffi = require 'ffi'
local threads = require 'threads'
M={}


function M.generateIndex(path, dataset)
	local size={}
  for i,d in pairs(dataset) do
    print('Generating Index', d)
    local train = path .. d .. '_train_lmdb'
    --local val = path .. dataset .. '_lmdb_val'  
    local value = {train} --,val}
    
    for i,v in pairs(value) do      
      db = lmdb.env{Path=v, RDONLY=true}
      db:open()
      reader = db:txn(true)
      cursor = reader:cursor()
      hash = tds.hash()
      count = 1
      local stop = false

      while not stop do
        local key,data = cursor:get()
      --print('Reading: ', count, 'Key:', key)
        hash[count] = key       
        count = count + 1

        if not cursor:next() then
          count = count - 1
          stop = true
        end
      end

      hash2 = torch.CharTensor(#hash, #hash[1])
      
      for i=1,#hash do 
        ffi.copy(hash2[i]:data(), hash[i], #hash[1]) 
      end
      file = v .. '_hashmaps.t7'
      torch.save(file, hash2)

      print('File: ' .. file .. ' saved, Size: ' .. count)
      table.insert(size,count)
    end    
  end
   	return size
end

function M.loadTensor(path,dataset)
	local size={}
   	local train = path .. dataset .. '_train_lmdb'
   	local val = path .. dataset .. '_val_lmdb'

   	local value = {train,val}

   	for i,v in pairs(value) do
   		print('Generating Index', v)
   		
   		table.insert(size,count)
   	end
   	return size
end

function M.cropImages(datapath)
  for f in paths.files(datapath, function(nm) return nm:find('.jpg') end) do
      local f2 = paths.concat(datapath, f)
      local im = image.load(f2)
      local x1, y1 = 30, 40
      local cropped = image.crop(im, x1, y1, x1 + 138, y1 + 138)
      local scaled = image.scale(cropped, 64, 64)
      image.save(f2, scaled)
      print("Saved: ",f2)
  end
end

function M.getClasses(path)
  directories = paths.dir(path)
  print(path,directories)
  local classes = {}
  for i,v in pairs(directories) do
    if v:find('img_') then
      table.insert(classes,v)
    end
  end
  return classes
end

function M.getdataSize(datapath)
  files = paths.dir(datapath)
  local size = 0
  for i,v in pairs(files) do
    if v:find('.jpg') then
      size = size+1
    end
  end
  return size
end

function M.getImages(datapath)
  images = {}
  files = paths.dir(datapath)
  --print("Files: ",# files)
  local size = 0
  for i,v in pairs(files) do
    if v:find('.jpg') then
      table.insert(images,v)
    end
  end
  return images
end

return M
