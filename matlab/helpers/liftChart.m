function [ V ] = liftChart( charts , V_flat , origin , corner)
%liftVertices lift given 
%   Inputs:
%       charts    - sz x sz x d  map that defines the embedding in R^d
%       V_flat    - #V x 2 embedding of vertices in 2D
%   Output:
%       V         - #V x d embedding of the vertices in R^d


% infer sizes
sz = size(charts,1);
d = size(charts,3);

% extract coordinates in 2D (u,v)
v = mod(V_flat(:,1)-origin, corner-origin) + origin;
u = mod(V_flat(:,2)-origin, corner-origin) + origin;

% define grid
[X,Y] = meshgrid(linspace(origin,corner,sz),linspace(origin,corner,sz));

% interpolate
V = zeros(size(v,1),d);
for ii = 1:d
    V(:,ii) = interp2(X,Y,squeeze(charts(:,:,ii)),v,u);
end

end

