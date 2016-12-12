require 'lmdb'
require 'image'
tds=require 'tds'
--ffi = require 'ffi'
local threads = require 'threads'
--threads.serialization('threads.sharedserialize')
M={}

function M.createThreads(nThreads,path,dataset)
	print("Create Threads")
    local pool = threads.Threads(nThreads,
                             function() 
                             	require 'torch' 
                             	require 'lmdb'
                              ffi = require 'ffi'
                              local test =5
                              datasize = {}
                              loadData = {}
                              local function loadImage(blob)
  --print("BLOB: ",blob)
   local input = image.decompress(blob, 3, 'float')
   --print("Size1",#input,#blob)

   -- find the smaller dimension, and resize it to loadSize[2] (while keeping aspect ratio)
   local iW = input:size(3)
   local iH = input:size(2)
   if iW < iH then
      input = image.scale(input, 32, 32 * iH / iW)
   else
      input = image.scale(input, 32 * iW / iH, 32)
   end
   return input
end


local trainHook = function(path)
   local input = loadImage(path)
   local iW = input:size(3)
   local iH = input:size(2)

   -- do random crop
   local oW = 16;
   local oH = 16
   local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
   local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
   local out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
   assert(out:size(2) == oW)
   assert(out:size(3) == oH)
   -- do hflip with probability 0.5
   if torch.uniform() > 0.5 then out = image.hflip(out); end
   out:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]
   return out
end


function getlsunSample(idx)
  local hash = ffi.string(loadData[1][1][idx]:data(), loadData[1][1]:size(2))
  img = loadData[1][3]:get(hash)
  local out = trainHook(img)
  return out
end

function getrealLabel(idx)
    return torch.LongTensor{1}
end
                             end,
                             function(idx)
                                threadid = idx
                                                                
                                for i,v in pairs(dataset) do
                                	print("Loading Dataset: ",v)
                                	loadData[i] = {}
                                	datasize[i] = {}
                                	for j=1,2,3 do
                                		loadData[i][j]={}
                                	end
                                	--print(path..v..'_train_lmdb_hashes_chartensor.t7')
                                	loadData[i][1] = torch.load(path..v..'_train_lmdb_hashes_chartensor.t7')
                                	--print("LoadData ",loadData[i][1])
                                	datasize[i] = loadData[i][1]:size(1)
                                  loadData[i][2] = 	lmdb.env{Path = path..v..'_train_lmdb',
                                				RDONLY=true, NOLOCK=true, NOTLS=true, NOSYNC=true, NOMETASYNC=true,
                               					MaxReaders=20, MaxDBs=20}
                               		loadData[i][2]:open()
                               		loadData[i][3] = loadData[i][2]:txn(true)
                                end
                                local hash = ffi.string(loadData[1][1][1]:data(), loadData[1][1]:size(2))
                                img = loadData[1][3]:get(hash)
                                print("Hash",hash, #img)
                                print("Starting thread",threadid)
                             end

    )
   
   return pool
end

function M.generateIndex(path,dataset)
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
		    if not cursor:next() then
		    	count = count - 1
		    	stop = true
		    end
		end

		hash2 = torch.CharTensor(#hash, #hash[1])
		for i=1,#hash do 
			ffi.copy(hash2[i]:data(), hash[i], #hash[1]) 
		end
   		
   		torch.save(v .. '_hashes_chartensor.t7', hsh2)
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

return M