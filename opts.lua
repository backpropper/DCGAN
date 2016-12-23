local M = {}


function M.getOpt()
    opt = {
        datatype = 'celeb',                                                  -- lsun / celeb
        batchSize = 64,
        inpdim = 100,                                                       -- Dim for input noise: inpdim x 1 x 1
        ngenfil = 64,                                                       -- Number of generator filters in first convolution layer
        ndisfil = 64,                                                       -- Number of discriminator filters in first convolution layer
        nThreads = 4,                                                       -- Number of Threads used in parallel iterator
        niter = 25,                                                         -- Number of epochs for training
        lr = 0.0002,                                                        -- Learning rate for adam
        beta1 = 0.5,                                                        -- momentum for adam
        gpu = false,                                                        -- gpu = false is CPU mode. gpu=true is GPU mode on
        name = 'Exp1',                                                      -- name of the experiment
        noise = 'normal',                                                   -- Noise type: uniform / normal
        generateIndex = 'false',                                            -- For LSUN data, if hash Index not generated leave it as true
        path = '/scratch/ag5799/cv_project/celeba/', 			            -- folder location for the lsun / celeb data 
        dataset = {'church_outdoor'},                                       -- if using lsun datatype- mention the dataset type(only one)
                                                                            -- {'bedroom', 'bridge', 'church_outdoor', 'classroom', 'conference_room',
                                                                            -- 'dining_room', 'kitchen', 'living_room', 'restaurant', 'tower'}
        debug = true,                                                       -- For debugging
        cropImages = false,                                                 -- cropping celebA images and rescaling to 64x64- see utils.lua-cropImages()
        imageSize = 64,                                                    -- Output Image Size: 64/128/256   
    }
    print(opt)
    return opt
end
    
return M
