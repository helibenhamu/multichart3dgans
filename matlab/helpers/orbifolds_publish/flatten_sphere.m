function [V_flat,cutMesh,flattener]=flatten_sphere(V,T,inds,orbifold_type,varargin)
p = inputParser;
p.addParameter('CutMesh',[])
p.addParameter('classic',false)
p.addParameter('verbose',true,@islogical)
p.addParameter('minimize_conformal',false,@islogical);
orbifold_type=uint32(orbifold_type);

%the cone structure of the 4 orbifolds, defined by the cone angles at the
%1st, 3rd, and possibly 4th cone.
singularities={[4 4], %type I
               [3 3], %type II
               [6 2], %type III
               [2 2 2]}; %type IV
           
if orbifold_type<4 && orbifold_type>=1
    if length(inds)~=3
        error('type I-III orbifolds requires 3 cones');
    end
elseif orbifold_type==4
    if length(inds)~=4
        error('type IV orbifold requires 4 cones');
    end
else
    error('orbifold type should be an integer between 1 and 4');
end
M_orig=[];
M_orig.V=V';
M_orig.F=T';
p.parse(varargin{:});
if isempty(p.Results.CutMesh)
    flattener=Flattener(M_orig,inds,singularities{orbifold_type});
else
    flattener=Flattener(M_orig,inds,singularities{orbifold_type},p.Results.CutMesh);
end
flattener.verbose=p.Results.verbose;
if p.Results.classic
    flattener.flatten('square');
else
    flattener.flatten();
end
if p.Results.minimize_conformal
    assert(orbifold_type==4);
    flattener.correctGlobalAffine;
end
V_flat=flattener.flat_V;
cutMesh=flattener.M_cut;

end