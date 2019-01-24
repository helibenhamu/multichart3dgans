function [ M,t ] = computeSimilarity( p1,p2,q1,q2 )
%Compute the similarity affine transformation T s.t T(p_i)=q_i, i=1,2
%Output: the similarity matrix M and the translation t

% a   -b    pi_1 + t1     qi_1
%         *            = 
% b    a    pi_2 + t2     qi_2

%so the linear equation to solve:
%pi_1*a+(-pi_2)*b+t1=pi_1
%pi_1*b+pi_2*a+t2=pi_2

%ordering of variables in vectorization: (a,b,t1,t2)

A=zeros(4,4);

A=[p1(1) -p1(2) 1 0;
   p2(1) -p2(2) 1 0;
   p1(2) p1(1) 0 1;
   p2(2) p2(1) 0 1];
rhs=[q1(1);q2(1);q1(2);q2(2)];

sol=A\rhs;
M=[sol(1) -sol(2);sol(2) sol(1)];
t=[sol(3);sol(4)];

d=M*[p1' p2']-[q1' q2']+[t t];
assert(max(abs(d(:)))<1e-12);

end

