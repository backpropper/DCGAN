local M = {}

function M.parse(arg)
    local cmd = torch.CmdLine();
    cmd:text()
    cmd:text('::::::DCGAN::::::')
    cmd:text()
    cmd:text('Options:')
    cmd:option('-dataset',         'lsun',             'Dataset to load')
    cmd:option('-batchSize',         64,             'Batch Size')
    cmd:option('-fineSize',          64,             'fineSize')
    cmd:option('-loadSize',          96,            'LoadSize')
    cmd:option('-inpdim',           100,            'Dim for input noise')
    cmd:option('-ngenfil',          64,              'Number of gen filters in first convolution')
    cmd:option('-ndisfil',          64,            'Number of dis filters in first convolution')
    cmd:option('-nThreads',         4,            'Number of Threads')
    cmd:option('-niter',            25,            'Starting learning rate iterations')
    cmd:option('-lr',            0.0002,           'learning rate')
    cmd:option('-beta1',           0.5,             'Momentum for Adam')
    cmd:option('-ntrain',      math.huge,           '#  of examples per epoch. math.huge for full dataset')
    cmd:option('-display',          false,          'Display samples while training')
    cmd:option('-gpu',              false,          'gpu mode')
    cmd:option('-noise',          'normal',          'Noise type uniform/normal')
    cmd:option('-name',              'Exp',          'Experiment number')
    cmd:option('-generateIndex',     false,          'Generate Index')

    local opt = cmd:parse(arg or {})

    return opt
end

return M
