function flattenDataset(datasets_names, database_signature)
%flattenDataset Flattens the meshes in the given datasets
%   flattens all the meshes of the given datasets and saves the flattenings
%   at: ../databases/images/<database_signature>


load(['../databases/flattening_parameters/params_', database_signature]);

% create directories
images_path = fullfile('../databases/images', database_signature);
mkdir(images_path)     % flattenings folder

for jj = 1:length(datasets_names)
    dataset_name = datasets_names{jj};
    meshes_path = fullfile('../databases/meshes', dataset_name);
    load(fullfile('../databases/flattening_parameters/', dataset_name, 'uCones.mat'));
    params.triplets = uCones(params.triplets_table);

    DirList = dir(meshes_path);
    file_paths = {DirList.folder; DirList.name};
    file_paths = cellfun( @(x,y) fullfile(x, y), file_paths(1,:), file_paths(2,:), 'UniformOutput', false); 

    % flatten all meshes
    for ii = 1:length(file_paths)
        cur_file_path = file_paths{ii};
        [~, shortname, ext] = fileparts(cur_file_path);

        out_path = fullfile(images_path,[shortname,'_img.mat']);
        % if file already exists, don't flatten again
        if isfile(out_path)
            disp([out_path, ' already exists']);
            continue;
        end
        % additional supported file extensions could be added here
        switch ext
            case '.obj'
                [V, F] = read_obj(cur_file_path);
                V=V'; F=F';
            case '.off'
                [V,F] = read_off(cur_file_path);
                V=V'; F=F';
            otherwise 
                warning(['file ', cur_file_path, ' is not of a valid file type']); 
                continue;
        end

        % flatten current mesh
        disp(['---- flattening ' shortname ,' ----'])
        [data, ~] = flattenMesh(V, F, params);

        % reduce mean and permute dimensions to be compatible with python 
        MeanVec = mean(mean(data,1),2);
        data = colStack(permute(bsxfun(@minus, data, MeanVec),[ 2 1 3 ]))';

        if sum(isnan(data(:)))==0
            save (out_path,'data');
        else
            warning(['file ', cur_file_path, ' failed to be flattened']);
        end
    end
end

