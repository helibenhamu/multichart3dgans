%% =====================================================================
%  ===                      Setting up stuff                         ===
%  =====================================================================
rng(0)
%Read the mesh
% [V,T]=read_off('sphere.off');V=V';T=T';
% load 12.mat;
[V,T]=read_off('k06_sas.off');V=V';T=T';
%Some color-related variables - no need to concern yourself with these :)
cone_colors=[1 0.8 0;0.7 0 1; 0 0.5 0.8;0 0 0.5];
cut_colors=[1 0 0;0 1 0;0 0 1];

%The cone positions and the choice of orbifold structure, defining the
%desired embedding

% === TYPE I orbifold ===
% cones=[50 100 200];
tri=triangulation(T,V);
inner=setdiff(1:length(V),tri.freeBoundary());
inds=inner(randsample(1:length(inner),3));
cones=inds([1 3 2]);
orbifold_type=OrbifoldType.Square;

%try uncommneting each of the next lines for the other orbifold structures

% === TYPE II orbifold ===
% cones=[50 100 130];
% orbifold_type=OrbifoldType.Diamond;

% === TYPE III orbifold ===
% cones=[50 100 130];
% orbifold_type=OrbifoldType.Triangle;

% === TYPE IV orbifold ===
% cones=[50 100 130 140];
% orbifold_type=OrbifoldType.Parallelogram;





%% =======================================================================
%  =======       The actual algorithm! cutting and flattening      =======
%  =======================================================================


%HAGGAI - Last two arguments are what you need to add

[V_flat,cutMesh,f]=flatten_sphere(V,T,cones,orbifold_type);%,'classic',true);





%% =======================================================================
%  =====                        Visualization                        =====
%  =======================================================================



figure(2);
clf;


% === Visualization of the original 3D mesh + cones & cuts on it ===
subplot(1,2,1);
%draw the mesh
patch('faces',T,'vertices',V,'facecolor','interp','FaceVertexCData',V(:,1),'edgecolor','k');
hold on;
%draw the cones
for i=1:length(cones)
    scatter3(V(cones(i),1),V(cones(i),2),V(cones(i),3),80,cone_colors(i,:),'fill');
end
%draw the cuts...
%iterate over all path-pairs (the correspondences generated in the
%cutting)
for i=1:length(cutMesh.pathPairs)
    %take the current path
    curPathPair=cutMesh.pathPairs{i};
    %use the mapping of indices of the cut mesh to the indices of the uncut 
    %mesh in order to get the cuts on the uncut mesh
    curPath=cutMesh.cutIndsToUncutInds(curPathPair(:,1));
    %draw the cut
    line(V(curPath,1),V(curPath,2),V(curPath,3),'color',cut_colors(i,:),'linewidth',2);
end
%some nice lighting and fixing the axis
camlight
axis equal




% === Visualization of the original 3D mesh + cones & cuts on it ===
subplot(1,2,2);
%draw the flattened mesh 
patch('faces',cutMesh.T,'vertices',V_flat,'facecolor','interp','FaceVertexCData',V(cutMesh.cutIndsToUncutInds,1),'edgecolor','k');
hold on;

%draw the cuts...
%iterate over all pairs of corresponding indices of twin vertices generated 
%in the cutting process
for i=1:length(cutMesh.pathPairs)
    %get the current pair of paths, corresponding to a single cut
    %connecting two cones
    curPathPair=cutMesh.pathPairs{i};
    %iterate over the two sides of the cut
    for j=1:2
        %get the current path (part of the boundary of the cut mesh between
        %two cones)
        curPath=curPathPair(:,j);
        %draw the line with the correspodning color
        line(V_flat(curPath,1),V_flat(curPath,2),'color',cut_colors(i,:),'linewidth',2);
    end
end
%draw the cones
for i=1:length(cones)
    %for each cone, find all its copies using the mapping of vertex indices 
    %of the uncut mesh to the vertex indices of the cut mesh
    flat_cones=cutMesh.uncutIndsToCutInds{cones(i)};
    scatter(V_flat(flat_cones,1),V_flat(flat_cones,2),40,cone_colors(i,:),'fill');
end
axis equal

