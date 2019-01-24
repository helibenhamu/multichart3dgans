function calculateAlignmentParams(database_signature)

% chart alignment params
% this script creates and saves the chart alignment parameters
%
%   The parameters are the ones needed in order to solve the
%   LS problem: Aw = b
%                   --- w, the variables vector ---
%                   w is of size 4*#charts - the 4 if for the degrees of
%                   freedom of each chart scale s (scalar) and translation
%                   t (3 dof).
%                   x is of the form: x = [ s1 t1x t1y t1z s2 t2x t2y ..]'
%
%                   --- A defines the equations ---
%                   First 4 equations fix a chart:
%                   si = 1, ti = 0
%                       [ ... 1 0 0 0 ... ;        [ 1
%                         ... 0 1 0 0 ... ;  * w =   0
%                         ... 0 0 1 0 ... ;          0
%                         ... 0 0 0 1 ... ]          0 ]
%                   For each pair of coinciding cones (i,j),
%                   3 equations are defined in the following
%                   form::
%                       [ ... xi 1 0 0 ... -xj -1  0  0 ... ;        [ 0
%                         ... yi 0 1 0 ... -yj  0 -1  0 ... ;  * w =   0
%                         ... zi 0 0 1 ... -zj  0  0 -1 ...  ]         0 ]
%                   (*) if regularization on the scale is also added,
%                   additional #charts equations will be added to the end
%                   of A with 1's in the columns corresponding to the scale
%                   variables. (those will be multiplied by the
%                   regularization parameter in the network code in tensorflow).
%
%   If regularization of the scale is added, also calculate the STD of the
%   dataset.
% --------------------------------------------------------------------------
%       triplets_mat         - the part of A that does not depend on the
%                              values of the cones. i.e only the blocks of
%                              ones and minus ones.
%                                   ith block     jth block
%                              [ ... 0 1 0 0 ... 0 -1  0  0 ... ;
%                                ... 0 0 1 0 ... 0  0 -1  0 ... ;
%                                ... 0 0 0 1 ... 0  0  0 -1 ...  ]
%
%       gather_indices_m,    - indices to extract the cones
%       gather_indices_p       values from the charts tensor.
%
%       scatter_indices_m,  - indices to place the
%       scatter_indices_p     extracted values of the cones in
%                             the triplets_mat (A).
%
%       triplets_masks       - ( |uCones| x #triplets x 3 )  boolean masks
%                              of positions of a cone in the triplets table
%
%       STD                  - the mean STD of the dataset
% ---------------------------------------------------------------------------

%% output path
out_path = fullfile('../databases/tfrecords', database_signature);
mkdir(out_path)

%% load triplets
load(['../databases/flattening_parameters/params_', database_signature]);
default_fixed_chart = 1;
num_of_parameters =  size(params.triplets_table,1)*4;
% define triplets and unique cones
triplets = params.triplets_table;  % #triplets x 3
uCones = unique(triplets);  % # of unique cones in triplets

%% Find coinciding cones in different triplets
triplet_pairs = [];
point_pairs = [];

for ii = 1:length(uCones)
    flags = find(triplets == uCones(ii));
    if length(flags)>1
        [R ,C] = ind2sub(size(triplets),flags);
        triplet_pairs = [triplet_pairs ; nchoosek(R,2)]; %#ok<AGROW>
        point_pairs = [point_pairs ; nchoosek(C,2)];    %#ok<AGROW>
    end
end

%% indices for python

gather_indices_p = int32([colStack(repmat(point_pairs(:,1),1,3)'-1 ), colStack([3*(triplet_pairs(:,1)-1),3*(triplet_pairs(:,1)-1)+1, 3*(triplet_pairs(:,1)-1)+2]')]);
gather_indices_m = int32([colStack(repmat(point_pairs(:,2),1,3)'-1 ), colStack([3*(triplet_pairs(:,2)-1),3*(triplet_pairs(:,2)-1)+1, 3*(triplet_pairs(:,2)-1)+2]')]);

scatter_indices_p = int32([ 4+((1:3*size(triplet_pairs,1))-1)' , colStack(repmat(4*(triplet_pairs(:,1)-1),1,3)')]);
scatter_indices_m = int32([ 4+((1:3*size(triplet_pairs,1))-1)' , colStack(repmat(4*(triplet_pairs(:,2)-1),1,3)')]);

triplets_mat = zeros(3*size(triplet_pairs,1),4*size(triplets,1));

for ii = 1:size(triplet_pairs,1)
    triplets_mat(3*(ii-1)+1:3*ii,(4*(triplet_pairs(ii,1)-1)+2):4*triplet_pairs(ii,1)) = eye(3);
    triplets_mat(3*(ii-1)+1:3*ii,(4*(triplet_pairs(ii,2)-1)+2):4*triplet_pairs(ii,2)) = -eye(3);
end
triplets_mat = sparse(triplets_mat);

triplets_mat = [ circshift([eye(4) , zeros(4,num_of_parameters-4)], default_fixed_chart*4, 2) ; triplets_mat];
save(fullfile(out_path, 'triplets_mat'),'triplets_mat');
B = [];
r = [zeros(1,num_of_parameters-1), 1];
for i =1:(num_of_parameters/4) , B = [B ; circshift(r,(i-1)*4+1,2)]; end
triplets_mat = [ triplets_mat ; B];
save(fullfile(out_path, 'triplets_mat_reg'),'triplets_mat');
save(fullfile(out_path, 'gather_indices_p'),'gather_indices_p');
save(fullfile(out_path, 'gather_indices_m'),'gather_indices_m');
save(fullfile(out_path, 'scatter_indices_p'),'scatter_indices_p');
save(fullfile(out_path, 'scatter_indices_m'),'scatter_indices_m');

for ii = 1:length(uCones)
    triplets_masks(ii,:,:) = (triplets==ii)';
end

save(fullfile(out_path, 'triplets_masks'),'triplets_masks');


images_path = fullfile('../databases/images', database_signature);
files = dir([images_path, '/*.mat']);
for ii = 1:length(files)
    load(fullfile(files(ii).folder, files(ii).name));
    data = reshape(data, params.sz^2*3,size(triplets,1));
    v(ii,:) = var(data);
end
STD = sqrt(mean(v));
save(fullfile(out_path, 'data_std'),'STD')
end