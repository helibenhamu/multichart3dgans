function [dataFunctions,flatteningInfo] = pushFunctionsToParameterization( V, F, cones, functions, params)

params.null = [];
doplot = getoptions(params,'doplot_flattened',0);
addScale = getoptions(params,'addScale',0);
stop_flag = getoptions(params, 'stop_flag', 0);
method = getoptions(params, 'method', 'e_interp');

% for flattening the mesh
orbifold_type=OrbifoldType.Square;
[V_flat,cutMesh,flattener]=flatten_sphere(V,F,cones,orbifold_type);

flatteningInfo = struct('V_flat',V_flat,'cutMesh',cutMesh,'flattener',flattener);
dataFunctions =  [];

if stop_flag
    return 
end

if addScale
    functions = [functions, flatteningInfo.flattener.valsOnUncutMesh(flattener.vertexScale())];   
end

numFunctions = size(functions,2);

% functions
if strcmp(method,'e_interp')
    sz = params.sz;
    f=functions(:,1);
    [V_flat_merged, T_merged] = make_tiling_interp(V_flat, cutMesh.T);
    vals = repmat(cutMesh.cutIndsToUncutInds,1,36);
    T_merged = T_merged';
    tt=zeros(3,size(T_merged,2));
    tt(:)=V_flat_merged(1,T_merged(1:3,:));
    xmin=min(tt);
    xmax=max(tt);
    tt(:)=V_flat_merged(2,T_merged(1:3,:));
    ymin=min(tt);
    ymax=max(tt);    
    ind_xmin = xmin>-4;
    ind_xmax = xmax<2;
    ind_ymin = ymin>-4;
    ind_ymax = ymax<2;
    
    ind = ind_xmin & ind_xmax & ind_ymin & ind_ymax;
    T_merged = T_merged(:,ind);  
    
    X = linspace(-3,1-4/sz,sz);
    Y = linspace(-3,1-4/sz,sz);    
    [out,tn,al2,al3]=mytri2grid(V_flat_merged,T_merged,f(vals),X,Y);
    dataFunctions(:,:,1) = out;

    for ii=2:numFunctions
        f=functions(:,ii);
        [out,tn,al2,al3] = mytri2grid(V_flat_merged,T_merged,f(vals),tn,al2,al3);
        dataFunctions(:,:,ii) = out;
    end    
    
else
    tic
    for ii=1:numFunctions
        f=functions(:,ii);
        [fOnFlatGrid ,~,~,~] =  captureFunctionOnTriangles(V_flat,cutMesh,f,params);
        dataFunctions(:,:,ii) = fOnFlatGrid;
    end
    toc
end

% plot flattened images
if doplot
    for ii=1:numFunctions
        figure, imagesc(dataFunctions(:,:,ii))
    end
end

end


