
local M = {}

local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias = nil
      m.gradBias = nil
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

function M.generator(noise_input, dis_filters, gen_filters, output)
	
	local BatchNormalization = nn.SpatialBatchNormalization
	local Convolution = nn.SpatialConvolution
	local FullConvolution = nn.SpatialFullConvolution

	local gen = nn.Sequential()
	gen:add(FullConvolution(noise_input, gen_filters * 8, 4, 4))
	gen:add(BatchNormalization(gen_filters * 8))
	gen:add(nn.ReLU(true))
	
	gen:add(FullConvolution(gen_filters * 8, gen_filters * 4, 4, 4, 2, 2, 1, 1))
	gen:add(BatchNormalization(gen_filters * 4))
	gen:add(nn.ReLU(true))
	
	gen:add(FullConvolution(gen_filters * 4, gen_filters * 2, 4, 4, 2, 2, 1, 1))
	gen:add(BatchNormalization(gen_filters * 2))
	gen:add(nn.ReLU(true))
	
	gen:add(FullConvolution(gen_filters * 2, gen_filters, 4, 4, 2, 2, 1, 1))
	gen:add(BatchNormalization(gen_filters))
	gen:add(nn.ReLU(true))
	
	gen:add(FullConvolution(gen_filters, output, 4, 4, 2, 2, 1, 1))
	gen:add(nn.Tanh())


	gen:apply(weights_init)
	return gen
end

function M.discriminator(noise_input, dis_filters, gen_filters, output)
	
	local BatchNormalization = nn.SpatialBatchNormalization
	local Convolution = nn.SpatialConvolution
	local FullConvolution = nn.SpatialFullConvolution

	local dis = nn.Sequential()

	dis:add(Convolution(output, dis_filters, 4, 4, 2, 2, 1, 1))
	dis:add(nn.LeakyReLU(0.2, true))

	dis:add(Convolution(dis_filters, dis_filters * 2, 4, 4, 2, 2, 1, 1))
	dis:add(BatchNormalization(dis_filters * 2))
	dis:add(nn.LeakyReLU(0.2, true))

	dis:add(Convolution(dis_filters * 2, dis_filters * 4, 4, 4, 2, 2, 1, 1))
	dis:add(BatchNormalization(dis_filters * 4))
	dis:add(nn.LeakyReLU(0.2, true))

	dis:add(Convolution(dis_filters * 4, dis_filters * 8, 4, 4, 2, 2, 1, 1))
	dis:add(BatchNormalization(dis_filters * 8))
	dis:add(nn.LeakyReLU(0.2, true))

	dis:add(Convolution(dis_filters * 8, 1, 4, 4))
	dis:add(nn.Sigmoid())

	dis:add(nn.View(1):setNumInputDims(3))

	dis:apply(weights_init)
	return dis
end

return M