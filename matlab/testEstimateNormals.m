%% point normal estimation using pca
clear;
cuda = Cuda();

pts=randn(3,10);
neighRadius=0.025;

for i=1:10
    k=1;
    for j=1:10
        diff=pts(:,i)-pts(:,j);
        if(norm(diff)<neighRadius)
            neighbors(:,k)=pts(:,j)
            k=k+1;
        end
    end
    A = cov(neighbors); %is always symmetric(self adjoint) and positive semi definite
    [vec,val] = eig(A);
    nor(:,i)=vec(:,3);
end



normals = cuda.estimateNormals(pts,neighRadius);

e_l(i)=err(l,lC,'3x3 l');