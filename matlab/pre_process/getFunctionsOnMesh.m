function functionsOnMesh = getFunctionsOnMesh(V,F,params)

doplotfunctions = getoptions(params,'doplotfunctions',0);

if isempty(params.func)
    functionsOnMesh = V;
else
    functionsOnMesh = params.func;
end


if doplotfunctions
    for ii = 1:size(functionsOnMesh,2)
        figure
        patch('faces',F,'vertices',V,'facecolor','interp','FaceVertexCData',functionsOnMesh(:,ii),'edgecolor','black'); axis off, axis equal
    end
end

end
