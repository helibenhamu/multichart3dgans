function BC = compute_map_from_sphere_embeddings( V_source,V_target,cutM_source,cutM_target )
% Given embeddings of two meshes, source and target, represented via V_*,T_*,
% and the segementation of their boundary according to their cones
% (boundary_*), compute the barycentric coordinates of all source vertices
% w.r.t the target mesh.
T_source=cutM_source.T;
T_target=cutM_target.T;
tiler=Tiler(V_source,T_source,cutM_source.pathPairs);
tiler.tile(6);

%find containing target triangle for each source vertex
tri=triangulation(T_target,V_target);

inds=1:length(V_source);
bc=zeros(length(V_source),3);
tri_ind=zeros(length(V_source),3);
for i=1:length(cutM_source.pathPairs)
    for k=1:2
    %current boundary segment
    p_target=cutM_target.pathPairs{i}(:,k);
    p_source=cutM_source.pathPairs{i}(:,k);
    
    % first handling the cones - which necessarily lie as end and start
    % vertices of the boundary segment
    
    startend_source=p_source([1 end]);
    startend_target=p_target([1 end]);
    
    % find any triangle containing the
    for j=1:2
        Tind=find(any(T_target==startend_target(j),2));
        assert(~isempty(Tind));
        Tind=Tind(1);
        tri_ind(startend_source(j))=Tind;
        bc(startend_source(j),:)=0;
        bc(startend_source(j),T_target(Tind,:)==startend_target(j))=1;
    end
    inds=setdiff(inds,startend_source);
    end
end
for i=1:length(tiler.trans)
    A=tiler.trans{i};
    V=V_source(inds,:)*A([1 2],:)'+repmat(A(3,:),length(inds),1);

     [cur_tri_ind,cur_bc]=tri.pointLocation(V);
     good=~isnan(cur_tri_ind);
     goodinds=inds(good);
     tri_ind(goodinds)=cur_tri_ind(good);
     bc(goodinds,:)=cur_bc(good,:);
     
     inds=inds(~good);
     
    if isempty(inds)
        break;
    end
    
end
if ~isempty(inds)
    error('some vertices not assigned');
end
reducedinds=cellfun(@(a)a(1),cutM_source.uncutIndsToCutInds);
bc=bc(reducedinds,:);
tri_ind=tri_ind(reducedinds);





%make sure all source vertices were matched
assert(~any(isnan(tri_ind)));

%these 3 lines are for making sure matrix has right size horizontal
%dimension
J=[cutM_target.cutIndsToUncutInds(T_target(tri_ind,:)); ones(1,3)*max(cutM_target.cutIndsToUncutInds)];
I=[1:length(bc) 1;1:length(bc) 1;1:length(bc) 1]';
V=[bc;0 0 0];
%create the barycentric coordinates matrix
BC=sparse(I,J,V);
end
