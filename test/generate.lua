require 'image'
require 'nn'
local optnet = require 'optnet'
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
    batchSize = 32,        -- number of samples to produce
    noisetype = 'normal',  -- type of noise distribution (uniform / normal).
    net = '',              -- path to the generator network
    noisemode = 'interpol',  -- random / interpol, for generating labels use random
    name = 'test/gen_images/generation',  -- name of the file saved
    gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
    nz = 100,
    outputType = 'multiple'  -- single/multiple, use single for producing single image 
    						-- for generating top labels
}

for k,v in pairs(opt) do 
	opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] 
end

print(opt)

if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
end

noise = torch.Tensor(opt.batchSize, opt.nz, 1, 1)
input = torch.Tensor(1, opt.nz, 1, 1)
net = torch.load(opt.net)

print(net)

if opt.noisetype == 'uniform' then
    noise:uniform(-1, 1)
elseif opt.noisetype == 'normal' then
    noise:normal(0, 1)
end
	
if opt.noisemode == 'interpol' then
    interpol  = torch.linspace(0, 1, opt.batchSize)
    firstpoint = torch.FloatTensor(opt.nz):uniform(-1, 1)
	secondpoint = torch.FloatTensor(opt.nz):uniform(-1, 1)
    for i = 1, opt.batchSize do
        noise:select(1, i):copy(firstpoint * interpol[i] + secondpoint * (1 - interpol[i]))
    end
end

if opt.gpu > 0 then
    net:cuda()
    cudnn.convert(net, cudnn)
    noise = noise:cuda()
    input = input:cuda()
end

if opt.outputType == 'single' then
	for i=1, opt.batchSize do
	    input:copy(noise[i])
	    local images = net:forward(input)
	    images:add(1):mul(0.5)
	    image.save(opt.name ..  tostring(i) .. '.png', image.toDisplayTensor(images))
	    print('Saved image to: ', opt.name .. tostring(i) .. '.png')
	end
	os.execute("python test/testimages.py")

elseif opt.outputType == 'multiple' then
	local images = net:forward(noise)
    print('Images size: ', images:size(1)..' x '..images:size(2) ..' x '..images:size(3)..' x '..images:size(4))
    images:add(1):mul(0.5)
    image.save(opt.name .. '.png', image.toDisplayTensor(images))
    print('Saved image to: ', opt.name .. '.png')
end

