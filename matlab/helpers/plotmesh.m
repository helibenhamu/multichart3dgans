function plotmesh(V,F,idx)
params.null = [];
colorVerts = V(:,idx)';
scattColor = bsxfun(@rdivide,bsxfun(@minus,colorVerts,min(colorVerts,[],1)), (max(colorVerts,[],1)-min(colorVerts,[],1)));
figure('Position', [100, 100, 800, 800]);

params.scattColor = scattColor;
params.verInd  = idx;
plotMeshAndPoints( V, F, params )

end


function  plotMeshAndPoints( V, F, params )

%--------------------------------------------
% Initialization
%--------------------------------------------
params.null = [];
n = getoptions(params,'n',1);
verInd = getoptions(params,'verInd',1:n);
scattColor = getoptions(params,'scattColor','c');
scattSize = getoptions(params,'scattSize',500);
labels = getoptions(params,'labels',{});

% check size of verInd compared to n
if n ~= size( verInd, 2 )
    warning('n = %d, size(verInd) = %d. Taking n to be %d.', n, size( verInd, 2 ), size( verInd, 2 ));
    n = size( verInd, 2 );
end
%============================================

%--------------------------------------------
% Take out NaNs
%--------------------------------------------
nanInd = isnan(verInd);
%============================================

%--------------------------------------------
% Extract verteces and faces information
%--------------------------------------------
axis equal; axis off;
if isempty(labels)
    labels = cellstr( num2str([1:n]') );
end


patch('vertices',V','faces',F','FaceColor',[0.6 0.6 0.6],'EdgeColor','k','FaceAlpha',1);
text(V(1,verInd(~nanInd)), V(2,verInd(~nanInd)), V(3,verInd(~nanInd)), labels(~nanInd), 'VerticalAlignment','bottom',...
            'HorizontalAlignment','right','FontSize',20,'FontWeight','bold','color','r');
hold on
scatter3(V(1,verInd(~nanInd)), V(2,verInd(~nanInd)), V(3,verInd(~nanInd)),scattSize,scattColor,'filled');

axis equal, axis off
addRot3D;
%============================================

end


