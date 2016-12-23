require 'lmdb'
require 'image'
require 'torch'
tds=require 'tds'
ffi = require 'ffi'
local threads = require 'threads'
--threads.serialization('threads.sharedserialize')
M={}


function M.generateIndex(path, dataset)
	local size={}
   	local train = path .. dataset .. '_train_lmdb'
   	local val = path .. dataset .. '_val_lmdb'

   	local value = {train,val}

   	for i,v in pairs(value) do
   		print('Generating Index', v)
   		db = lmdb.env{Path=v, RDONLY=true}
   		db:open()
   		reader = db:txn(true)
   		cursor = reader:cursor()
   		hash = tds.hash()
   		count = 1
   		local stop = false

   		while not stop do
      		local key,data = cursor:get()
      		print('Reading: ', count, '   Key:', key)
      		hash[count] = key		    
      		count = count + 1
        if count == 300 then
          --stop = true
        end

		    if not cursor:next() then
		    	count = count - 1
		    	stop = true
		    end
		end

		hash2 = torch.CharTensor(#hash, #hash[1])
		for i=1,#hash do 
			ffi.copy(hash2[i]:data(), hash[i], #hash[1]) 
		end
   		
   		torch.save(v .. '_hashes_chartensor.t7', hash2)
   		print('wrote index file with ' .. count .. ' keys')
   		table.insert(size,count)
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
local data = datapath .. '/img_align_celeba'
  for f in paths.files(data, function(nm) return nm:find('.jpg') end) do
      local f2 = paths.concat(data, f)
      local im = image.load(f2)
      local x1, y1 = 30, 40
      local cropped = image.crop(im, x1, y1, x1 + 138, y1 + 138)
      local scaled = image.scale(cropped, 64, 64)
      image.save(f2, scaled)
  end
end
return M