function B= minimize_global_lscm(As,areas)
%input - vector of differentials, sorted (a_i,b_i,c_i,d_i), vector of areas
%output - B which is the solution of the problem min LSCM(B*As) s.t. ||B||=1

%build LSCM functional
N=size(As,1)/4;

S=kron(sparse(1:length(areas),1:length(areas),sqrt(areas)),[1 0 0 -1; 0 1 1 0]);
F=S'*S;




%now, we are looking for the matrix [x y;z w] with norm=1 s.t. the matrix
%  D = [x y;  *[a b;
%       z w]    c d]
% has minimial lscm ----> D = [ xa+yc xb+yd; za+wc zb+wd ] that is
% a c 0 0   x
% b d 0 0 * y
% 0 0 a c   z
% 0 0 b d   w

stencil_inds=[1 1;1 2;2 1;2 2];
shifts=4*repmat((0:N-1)',1,size(stencil_inds,1))';
shifts=shifts(:);
linds=repmat(stencil_inds,N,1)+[shifts zeros(size(shifts))];
rinds=linds+2;
inds=[linds;rinds];
vals=[As;As];
G2A=sparse(inds(:,1),inds(:,2),vals);

L=full(G2A'*F*G2A);


[v,c]=eig(L);
[~,ind]=min(diag(abs(c)));
B=reshape(v(:,ind),2,2)'*sqrt(2);%multiplying by sqrt(2) so as to have same norm as eye(2)
[U,E,V]=closest_rotation(B);
B=E*V';
end

