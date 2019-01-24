function AGD = calcAgd(V,F,params)
% V,F mesh representation
% n number of points to choose from (e.g. 5 for human)
%params
smoothing_n_itersAGD = getoptions(params, 'smoothing_n_itersAGD', 10); % number of smoothing iterations for AGD
doplot = getoptions(params,'doplot',0);

% nor,alize shape by shape area 
area = CORR_calculate_area(F,V);
V = V/sqrt(area);

nV = max(size(V));
% calculate AGD
disp('Calculating pairwise distances...')
adj = triangulation2adjacency_change(F,V');
dist = graphallshortestpaths(adj,'directed',false);
diameter = max(dist(:));
% dist = rand(nV);
disp('Calculating face areas...')
% calculate triangle areas
faceAreas = computeSurfAreas(V,F);

% calculate 1-ring areas
disp('Calculating 1-ring areas...')
oneRingAreas = zeros(nV,1);
for ii = 1:nV
    ff = any(F'==ii); % indices of faces in the ii'th vertex 1-ring
    oneRingAreas(ii) = (1/3)*sum(faceAreas(ff));
end

% calculate AGD (average geodesic distance)
disp('Calculating and smooth AGD...')
AGD_raw = dist*oneRingAreas;
% smoothing
AGD = perform_mesh_smoothing(F,V,AGD_raw,struct('niter_averaging',smoothing_n_itersAGD));


end