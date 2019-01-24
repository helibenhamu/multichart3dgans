function [ BC_1to2,BC_2to1,V_flat1,V_flat2,segs1,segs2 ] = map_disks(V1,T1,cones1,V2,T2,cones2,varargin)
p = inputParser;
p.addParameter('verbose',true,@islogical)
p.parse(varargin{:});
if p.Results.verbose
    disp('=== Embedding mesh #1 ====')
end
[V_flat1,segs1]=flatten_disk(V1,T1,cones1,varargin{:});
if p.Results.verbose
    disp('=== Embedding mesh #2 ====')
end
[V_flat2,segs2]=flatten_disk(V2,T2,cones2,varargin{:});
if p.Results.verbose
    disp('=== Computing surface map 1->2 ====')
    tid=tic;
end
BC_1to2 = compute_map_from_disk_embeddings( V_flat1,T1,V_flat2,T2,segs1,segs2 );
if p.Results.verbose
    toc(tid);
    disp('=== Computing surface map 2->1 ====')
    tid=tic;
end
BC_2to1 = compute_map_from_disk_embeddings( V_flat2,T2,V_flat1,T1,segs2,segs1);
if p.Results.verbose
    toc(tid);
end

end

