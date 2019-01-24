function [ BC1to2,BC2to1 ] = map_spheres( V1,T1,V2,T2,cones1,cones2,orbifold_type,varargin )
p = inputParser;
p.addParameter('verbose',true,@islogical)
p.parse(varargin{:});
[V_flat1,cutMesh1]=flatten_sphere(V1,T1,cones1,orbifold_type,varargin{:});
[V_flat2,cutMesh2]=flatten_sphere(V2,T2,cones2,orbifold_type,varargin{:});
if p.Results.verbose
    fprintf('Extracting surface mapping 1->2 from orbifold embeddings: ');
    tid=tic;
end
BC1to2=compute_map_from_sphere_embeddings( V_flat1,V_flat2,cutMesh1,cutMesh2 );
if p.Results.verbose
    toc(tid);
    
end
if p.Results.verbose
    fprintf('Extracting surface mapping 2->1 from orbifold embeddings: ');
    tid=tic;
end
BC2to1=compute_map_from_sphere_embeddings( V_flat2,V_flat1,cutMesh2,cutMesh1 );
if p.Results.verbose
    toc(tid);

    
end
end

