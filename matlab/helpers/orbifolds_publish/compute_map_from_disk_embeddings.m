function BC = compute_map_from_disk_embeddings( V_source,T_source,V_target,T_target,boundary_source,boundary_target )
% Given embeddings of two meshes, source and target, represented via V_*,T_*, 
% and the segementation of their boundary according to their cones
% (boundary_*), compute the barycentric coordinates of all source vertices
% w.r.t the target mesh.

%find containing target triangle for each source vertex
tri=triangulation(T_target,V_target);
[tri_ind,bc]=tri.pointLocation(V_source);
%the above finds the barycentric coorinates of all interior vertices;
%boundary vertices are more tricky as they lie on edges of triangles and
%prone to numerical instability. Hence we project the vertices onto the
%infinite line supporting the boundary of the embedding...

%go over each boundary segment (an "edge" of the disk orbifold polygon)
for i=1:length(boundary_source)
    
    %current boundary segment
    p_target=boundary_target{i};
    p_source=boundary_source{i};
    
    % first handling the cones - which necessarily lie as end and start 
    % vertices of the boundary segment
    
    startend_source=[p_source([1 end])];
    startend_target=[p_target([1 end])];
    
    % find any triangle containing the 
    for j=1:2
        Tind=find(any(T_target==startend_target(j),2));
        assert(~isempty(Tind));
        Tind=Tind(1);
        tri_ind(startend_source(j))=Tind;
        bc(startend_source(j),:)=0;
        bc(startend_source(j),T_target(Tind,:)==startend_target(j))=1;
    end
    
    
    %mapping boundary vertices which are not cones
    %take the vector of the direction of the "infinite line" on which the
    %edge of the orbifold polygon lies (line connecting the start and end
    %cones)
    
    vv=V_target(p_target(end),:)-V_target(p_target(1),:);
    
    %we do not wish to compute the map for the two cones so remove them
    p_source=p_source(2:end-1);
    %take the position along the inifinite line vv of all boundary vertices
    %both on source and target
    line_distance_target=V_target(p_target,:)*vv';
    line_distance_source=V_source(p_source,:)*vv';
    %map each source vertex
    for j=1:length(line_distance_source)
        %find the two consecutive boundary vertices on target between which
        %source vertex lies (all projected onto the infinite line vv)
        ind_of_containing_segment=find(line_distance_source(j)>line_distance_target(1:end-1)&line_distance_source(j)<line_distance_target(2:end));
        assert(length(ind_of_containing_segment)==1);
        ind_of_containing_segment=ind_of_containing_segment(1);
        
        %compute the barycentric coordinates of source vertex w.r.t the two
        %found target vertices. barycentric coords of x in [a,b] is
        %(x-a)/(b-a)
        c=(line_distance_source(j)-line_distance_target(ind_of_containing_segment))/(line_distance_target(ind_of_containing_segment+1)-line_distance_target(ind_of_containing_segment));
        %now find the triangle which has the containing boundary segment as
        %its edge
        triind=find(sum(ismember(T_target,[p_target(ind_of_containing_segment) p_target(ind_of_containing_segment+1)]),2)==2);
        assert(length(triind)==1);
        
        %inserting the correct triangle and barycentric coordinates into
        %the list
        tri_ind(p_source(j))=triind;
        %take the vertex indices of the triangle and insert the barycentric
        %coordinates in the correct order according to them
        TT=T_target(triind,:);
        %first zero-out all of them
        bc(p_source(j),:)=0;
        
        %since vertex lies on the edge, one coordinate is left as zero,
        %other two are filled as c and 1-c
        bc(p_source(j),TT==p_target(ind_of_containing_segment))=1-c;
        bc(p_source(j),TT==p_target(ind_of_containing_segment+1))=c;
    end
end
%make sure all source vertices were matched
assert(~any(isnan(tri_ind)));

%these 3 lines are for making sure matrix has right size horizontal
%dimension
J=[T_target(tri_ind,:); ones(1,3)*length(V_target)];
I=[1:length(V_source) 1;1:length(V_source) 1;1:length(V_source) 1]';
V=[bc;0 0 0];
%create the barycentric coordinates matrix
BC=sparse(I,J,V);
end
