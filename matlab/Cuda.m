classdef Cuda < handle
    
    properties
        k
        k2
        k3
        k4
        k5
    end
    
    methods
        function obj=Cuda()
         obj.k = parallel.gpu.CUDAKernel('../build/eig.ptx','../src/eig.cu','eigVal');
         
         obj.k2 = parallel.gpu.CUDAKernel('../build/eig.ptx','../src/eig.cu','eig');
         
         obj.k5 = parallel.gpu.CUDAKernel('../build/eig.ptx','../src/eig.cu','estimateNormals');
         
         obj.k3 = parallel.gpu.CUDAKernel('../build/cov.ptx','../src/cov.cu','cov');
         
         obj.k4 = parallel.gpu.CUDAKernel('../build/cov.ptx','../src/cov.cu','mean');
        end
        
        function [o1, o2] = eig(obj,A,varargin)
            n = size(A,1);
            if nargout == 1
                o1=zeros(n,1); %column vector of eigenvalues
                [o1] = feval(obj.k,A,o1,n);
            else
                o1=zeros(n); %matrix of right eigenvectors
                o2=zeros(n); %diagonal matrix of right eigenvalues
                useIterative=false;
                if(nargin>2)
                    useIterative=varargin{1};
                end
                [o1,o2] = feval(obj.k2,A,o1,o2,n,useIterative);
            end
        end
        
        function [C] = cov(obj,pts)
            [rows,cols]=size(pts);
            if(rows>cols)
                pts=pts';
            end
            [rows,cols]=size(pts);
            if(rows ~= 3)
                error('only float3 supported');
            end
            C=zeros(3,'single');
            [C]=feval(obj.k3,pts,C,cols);
    
        end
        
       function [m] = mean(obj,pts)
            [rows,cols]=size(pts);
            if(rows>cols)
                pts=pts';
            end
            [rows,cols]=size(pts);
            if(rows ~= 3)
                error('only float3 supported');
            end
            m=zeros(1,3,'single');
            [m]=feval(obj.k4,pts,m,cols);
    
       end
        
       function [m] = estimateNormals(obj,pts,neighRadius)
            [rows,cols]=size(pts);
            if(rows>cols)
                pts=pts';
            end
            [rows,cols]=size(pts);
            if(rows ~= 3)
                error('only float3 supported');
            end
            m=zeros(cols,3,'single');
            [m]=feval(obj.k5,pts,m,cols,neighRadius);
    
        end

    end
    
end

