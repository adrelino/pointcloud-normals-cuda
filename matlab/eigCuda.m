function [o1, o2] = eigCuda(A)
n = size(A,1);
if nargout == 1
    k = parallel.gpu.CUDAKernel('eig.ptx','eig.cu','eigVal');
    k.ThreadBlockSize = [1];
    k.GridSize = [1];
    
    o1=zeros(n,1); %column vector of eigenvalues
    [o1] = feval(k,A,o1,n);
else
    k2 = parallel.gpu.CUDAKernel('eig.ptx','eig.cu','eig');
    k2.ThreadBlockSize = [1];
    k2.GridSize = [1];
    
    o1=zeros(n); %matrix of right eigenvectors
    o2=zeros(n); %diagonal matrix of right eigenvalues
    [o1,o2] = feval(k2,A,o1,o2,n);

end

