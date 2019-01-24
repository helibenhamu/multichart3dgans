%% Post-Process - inspect generated meshes

% load dataset parameters
database_signature = 'humans_64x64_example';
load(['../databases/flattening_parameters/params_', database_signature]);

% load generated examples
load('data/generations_example.mat');

% load template mesh
[V_temp,F_temp] = read_ply('data/templateMesh.ply');
% to create fInfo file for a new template, simply run 
% [data, fInfo] = flattenMesh(V_temp, F_temp, params)
load(fullfile('data/fInfo_temp.mat'));

% reconstruct and show generated meshes
for sample = 1:5
charts = permute(squeeze(c(sample,:,:,:)) , [2,3,1]);
charts_aligned = align_charts_ST(charts,params.triplets_table,3);
[V_rec, ~, ~, ~] = reconstructMesh(charts, fInfo_temp, params.triplets_table);

figure
patch('vertices',V_rec,'faces',F_temp,'facecolor',[1 1 1],'edgecolor','none')
axis equal; addRot3D; axis off; camlight

end