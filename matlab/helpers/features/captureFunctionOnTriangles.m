function [out] =  captureFunctionOnTriangles(V_flat,cutMesh,f,params)

method = char(getoptions(params,'method','e_interp'));
interp_method = char(getoptions(params,'interp_method','linear'));
sz = getoptions(params,'sz',512);

if strcmp(method, 'patch')
    
    isseg = getoptions(params,'isseg',0);
    limits =  [-3,1, -3,1];
    h = figure('visible','off', 'Position', [100, 100, 1000, 1000]);
    make_tiling(V_flat, cutMesh, f,isseg);
    axis(limits)
    
    axis off
    set(gca,'units','pixels'); % set the axes units to pixels
    x = get(gca,'position'); % get the position of the axes
    set(gcf,'units','pixels'); % set the figure units to pixels
    y = get(gcf,'position'); % get the figure position
    set(gcf,'position',[y(1) y(2) x(3) x(4)]);% set the position of the figure to the length and width of the axes
    set(gca,'units','normalized','position',[0 0 1 1]); % set the axes units to pixel
    colormap(gray)
    
    frame = getframe(gca);
    out = imresize(frame.cdata(:,:,1),[sz sz]);
    out = double(out);
    out = (max(f(:))-min(f(:)))*(out-min(out(:)))/(max(out(:))-min(out(:)))+min(f(:));
    close(h);
    
elseif strcmp(method, 'interp')
    tic;
    [X,Y] = meshgrid(linspace(-3,1-4/sz,sz),linspace(-3,1-4/sz,sz));
    V_flat_merged = make_tiling_interp(V_flat, cutMesh.T);
    ind = ~((V_flat_merged(1,:)>-3.5 & V_flat_merged(1,:)<1.5) & (V_flat_merged(2,:)>-3.5 & V_flat_merged(2,:)<1.5));
    V_flat_merged(:,ind) = [];
    vals = repmat(cutMesh.cutIndsToUncutInds,1,36);
    vals(ind) = [];
    F = scatteredInterpolant(V_flat_merged(1,:)',V_flat_merged(2,:)',f(vals),interp_method,'nearest');
    out = F(X,Y);
    toc;
elseif strcmp(method, 'e_interp')
    X = linspace(-3,1-4/sz,sz);
    Y = linspace(-3,1-4/sz,sz);
    [V_flat_merged, T_merged] = make_tiling_interp(V_flat, cutMesh.T);    
    vals = repmat(cutMesh.cutIndsToUncutInds,1,36);
    [out,~,~,~] = mytri2grid(V_flat_merged,T_merged',f(vals),X,Y);     
else
    error('the method for capturing function on triangles in not valid. please choose either "patch" or "interp"')
end
end
