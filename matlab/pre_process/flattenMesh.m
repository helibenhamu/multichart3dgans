function [data, flatteningInfo] = flattenMesh(V, F, params)

params.null = [];
triplets = params.triplets;
stop_flag = getoptions(params, 'stop_flag', 0);
data =[];

% calculate the area of the surface
area = CORR_calculate_area(F,V);

% center and normalize the vertices so the area of the new surface will be 1
V = V-ones(size(V,1),1)*mean(V);
V = V/sqrt(area);

% get functions on the mesh
functionsOnMesh = getFunctionsOnMesh(V,F,params);

%for each triplet compute parameterization and push functions
for jj=1:size(triplets,1)
    disp(['Calculating flattening of triplets ', num2str(jj)])
    cones=triplets(jj,:);
    [dataFunctions{jj},flatteningInfo{jj}] = pushFunctionsToParameterization( V, F, cones, functionsOnMesh,params );
end

if stop_flag
    return 
end

% concatenate the features of the all the flattenings
dataFunctions_mat = reshape(dataFunctions, 1,1,size(triplets,1));
dataFunctions_mat = cell2mat(dataFunctions_mat);

% convert to single
data = single(dataFunctions_mat);
end