%% Pre-Process - create dataset of flattened meshes for learning
% -------------------------------------------------------------------------------
%  Before running this script the following requirements should be filled:
%
%       1. The meshes of the dataset should be in:
%          ../databases/meshes/<dataset_name>
%       2. Choose landmarks (sparse correspondence points) on the dataset
%          could use this tool for help: GUI_select_points(V,F)
%          save in:
%          ../databases/flettening_parameters/<dataset_name>/uCones.mat
%          uCones - 1x#landmarks , where the entries are indices of vertices.
%          when working with multiple datasets with the same landmarks,
%          save uCones for each (keeping the ordering of the indices).
%       3. Choose triplets of landmarks to define the charts while keeping
%          the rigidity requirement.
%          triplets_table - 3x#charts, indices in uCones of triplets of
%          cones.
%       4. Define parameters of database - 
%           params.sz - spatial resolution
%           params.func - functions to push to parametrization. 
%                         Default: if params.func = [] the x,y,z coordinates are
%                         pushed.
%           params.triplets_table - as in item 2. 
%           save in: 
%           ../databases/flettening_parameters/params_<database_signature>
%
% -------------------------------------------------------------------------------


% For this demo, you can find the requirements filled for 2 exmample
% datasets: example_dataset , example_dataset_2.
database_signature = 'humans_64x64_example';
datasets_names = {'example_dataset', 'example_dataset_2'};

% Flatten the meshes in the given datasets and save
% The data will be saved in: ../databases/images/<database_signature>
flattenDataset(datasets_names, database_signature)

% Calculate alignment parameters for the created database
% The alignment parametrs are saved in:
%   ../databases/tfrecords/<database_signature>/
% those parametersare used in the landmark consistency layer in the
% network.
calculateAlignmentParams(database_signature)

% To complete the preperation of the data for training, the data should be
% converted to tfrecords. To do so, run the script convert_to_tfrecords
% found in:
%   ../convert_to_tfrecords.py

