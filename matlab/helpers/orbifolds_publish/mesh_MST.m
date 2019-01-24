function [ tree ] = mesh_MST( T,edge,ignore )

E=[T(:,[1 2]);T(:,[1 3]);T(:,[2 3])];
E=sort(E,2);
E=unique(E,'rows');
A=sparse(max(max(T)),max(max(T)));
I=sub2ind(size(A),E(:,1),E(:,2));
A(I)=1;
A=A';
A(I)=1;

A(edge(1),:)=0;
A(:,edge(1))=0;
A(ignore,:)=0;
A(:,ignore)=0;
tree=myTree(A,edge(2));%graphminspantree(A,root);
tree(edge(1),edge(2))=1;
%now order tree correctly
% [i,j]=find(tree);
% %tree=[j i];
% I=sub2ind(size(A),j,i);
% A=sparse(max(max(T)),max(max(T)));
% A(i)=1;
%tree=tree';
end
function T=myTree(E,r)
% if ismember(r,mustBeLeaf)
%     return;
% end
% if nargin==3%init
T=sparse(length(E),length(E));
% end
stack=[r];
while ~isempty(stack)
    r=stack(1);
    stack(1)=[];    
%     if ismember(r,mustBeLeaf)
%         continue;
%     end
    
    sons=find(E(r,:));
    E(r,:)=0;
    E(:,r)=0;
    E(:,sons)=0;
    T(r,sons)=1;
    %[T,E]=myTree(E,sons(1),mustBeLeaf, T);
    stack=[stack sons];
    % end
end

end