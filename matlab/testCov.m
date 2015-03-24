n=10;
pts=randn(n,3,'single');
mGold=mean(pts)
CGold=cov(pts) %is always symmetric(self adjoint) and positive semi definite

m=single([0,0,0]);
for i=1:n
    m=m+pts(i,:); 
end
m=m/n;

C=zeros(3);
for i=1:n
    diff=pts(i,:)-m;
    C=C+diff'*diff;
end
C=C/(n-1);

C2=zeros(3);
for i=1:n
    diff=pts(i,:);
    C2=C2+diff'*diff;
end
C2=C2-(m'*m)*n;
C2=C2/(n-1);

% cuda = Cuda();
% 
% mCuda=cuda.mean(pts)
% CCuda=cuda.cov(pts)