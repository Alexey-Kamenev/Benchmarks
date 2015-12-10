require 'sys';
require 'bit';
require 'cunn';
require 'cudnn';
cudnn.benchmark = true;
cudnn.verbose = true;
require 'optim';
torch.setdefaulttensortype('torch.FloatTensor')

local steps = 1 -- number of runs

local Linear = nn.Linear
local Transfer = cudnn.ReLU
local hsize = 4096
local osize = 1000

-- Network definition
local cnn = nn.Sequential()
cnn:add(cudnn.SpatialConvolution(3,96,11,11,4,4,2,2)):add(Transfer(true))
cnn:add(cudnn.SpatialMaxPooling(3,3,2,2))
cnn:add(cudnn.SpatialConvolution(96,256,5,5,1,1,2,2)):add(Transfer(true))
cnn:add(cudnn.SpatialMaxPooling(3,3,2,2))
cnn:add(cudnn.SpatialConvolution(256,384,3,3,1,1,1,1)):add(Transfer(true))
cnn:add(cudnn.SpatialConvolution(384,384,3,3,1,1,1,1)):add(Transfer(true))
cnn:add(cudnn.SpatialConvolution(384,256,3,3,1,1,1,1)):add(Transfer(true))
cnn:add(cudnn.SpatialMaxPooling(3,3,2,2))

cnn:add(Linear(256*6*6,hsize)):add(Transfer(true)) -- hidden layer 1
cnn:add(nn.Dropout(0.5))
cnn:add(Linear(hsize,hsize)):add(Transfer(true)) -- hidden layer 2
cnn:add(nn.Dropout(0.5))
cnn:add(Linear(hsize,osize)):add(cudnn.LogSoftMax()) -- output layer

-- Fake data
local bsize = 256
local inputCPU = torch.randn(torch.LongStorage({bsize,3,224,224}))
local input = torch.CudaTensor(inputCPU:size())
local target = torch.IntTensor(bsize):random(1,osize):cuda()

for k=0,2 do
    nGPU = bit.lshift(1,k)

    local model = nil
    if nGPU > 1 then
        model = nn.DataParallelTable(1)
        for i=1,nGPU do
            cutorch.setDevice(i)
            model:add(cnn:clone():cuda(), i)
        end
        cutorch.setDevice(1)
    else
        model = cnn:cuda()
    end

    -- optimizer declarations
    local criterion = nn.ClassNLLCriterion():cuda()
    local parameters, gradParameters = model:getParameters()
    local optimState = { learningRate = 0.01 }

    collectgarbage()
    sys.tic()
    for t = 1, steps do
        input:copy(inputCPU) -- transfer data to GPU memory
        feval = function(x)
            model:zeroGradParameters()
            local output = model:forward(input)
            local err = criterion:forward(output, target)
            local gradOutput = criterion:backward(output, target)
            local gradInput = model:backward(input, gradOutput)
            return err, gradParameters
        end
        optim.sgd(feval, parameters, optimState)

        -- DataParallelTable's syncParameters
        model:apply(function(m) if m.syncParameters then m:syncParameters() end end)
        cutorch.synchronize()
    end
    local elapsed = sys.toc()

    print(string.format("%d GPUs: %0.0f samples per sec", nGPU, steps * bsize / elapsed))
end

