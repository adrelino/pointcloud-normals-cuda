%% 3x3 symmetric
clear;
cuda = Cuda();
A=gallery('lehmer',3);


for i=1:100
%     A=rand(3);
%     A(2,1)=A(1,2); %make symmetric
%     A(3,1)=A(1,3); %make symmetric
%     A(3,2)=A(2,3); %make symmetric
    
    pts=randn(10,3);
    A=cov(pts); %is always symmetric(self adjoint) and positive semi definite

    l = eig(A);
    lC = gather(cuda.eig(A));

    e_l(i)=err(l,lC,'3x3 l');

    [E,L] = eig(A);
    [EC,LC] = cuda.eig(A);
    EC=gather(EC);
    LC=gather(LC);

    e_L(i)=err(L,LC,'3x3 L');
    e_E(i)=err(E,EC,'3x3 E',true);
end

ml=mean(e_l)
mL=mean(e_L)
mE=mean(e_E)

%% 2x2 symmetric
clear;
cuda = Cuda();

A = [1 2; 2 4];

for i=1:100
    A=rand(2);
    A(2,1)=A(1,2); %make symmetric

    l = eig(A);
    lC = cuda.eig(A);

    e_l(i)=err(l,lC,'3x3 l');

    [E,L] = eig(A);
    [EC,LC] = cuda.eig(A);

    e_L(i)=err(L,LC,'3x3 L');
    e_E(i)=err(E,EC,'3x3 E',true);
end

ml=mean(e_l)
mL=mean(e_L)
mE=mean(e_E)