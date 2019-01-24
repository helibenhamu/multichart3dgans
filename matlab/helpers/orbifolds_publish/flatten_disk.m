function [V_flat,boundary_segments]=flatten_disk(V,T,inds,varargin)
p = inputParser;
p.addParameter('verbose',true,@islogical)




if length(inds)==3
    tri_or_square=true;
    
elseif length(inds)==4
    tri_or_square=false;
else 
    error('disk orbifolds need exactly 3 or 4 cones');
end
    

M_orig=[];
M_orig.V=V';
M_orig.F=T';
p.parse(varargin{:});

flattener=Flattener(M_orig,inds,[]);
flattener.verbose=p.Results.verbose;
if tri_or_square
    orbifold_type='freetri';
else
    orbifold_type='freesquare';
end
flattener.flatten(orbifold_type);

V_flat=flattener.flat_V;
boundary_segments=cellfun(@(X)X(:,1),flattener.M_cut.pathPairs,'UniformOutput',false);
end