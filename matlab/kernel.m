function k = kernel(fun,w,h,nc)
%KERNEL calls the specified kernel
%   grid size adapted to image size
if(nargin<4)
    nc=1;
end
g=gpuDevice;

bdX=32; %warp size, so we should have multiples of warp size

bdY=8;

bdZ=g.MaxThreadsPerBlock/(bdX*bdY); %is 4 if we can use 1024 threads per block
% if(bdZ>nc)
%     [num2str(bdX*bdY*(bdZ-nc)) ' unused threads per block']
%     bdZ=nc;
% end

pathSrc='eig';
pathPtx='eig';

%[path '::' fun]
k = parallel.gpu.CUDAKernel([pathPtx '.ptx'],[pathSrc '.cu'],fun);
k.ThreadBlockSize = [bdX,bdY,bdZ];
k.GridSize = [int32((w+bdX-1)/bdX),int32((h+bdY-1)/bdY),1]; %make the blocks overlap the image, but not the z dimension (here we need to make shure with a strided loop that we process all channels inside one z block only)

test1 = k.ThreadBlockSize > g.MaxThreadBlockSize;  %important: z dimension limit usually lower (64) than other limits
test2 = k.GridSize > g.MaxGridSize;

if(sum(test1)>0)
    error('threadBlockDimension not supported by device');
end

if(sum(test2)>0)
    error('gridDimension not supported by device');
end
end

