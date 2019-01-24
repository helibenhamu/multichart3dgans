function [Cmax, Cmin,Cmean, CsqrtGauss,Cgauss] = getCurvatureFeatures(V,F)
doplot = 0;
FV.vertices= V;
FV.faces = F;
[PrincipalCurvatures,PrincipalDir1,PrincipalDir2,FaceCMatrix,VertexCMatrix,Cmagnitude]= GetCurvatures( FV ,0);

Cmax = PrincipalCurvatures(1,:)';
Cmin = PrincipalCurvatures(2,:)';
Cmean = (PrincipalCurvatures(1,:)' + PrincipalCurvatures(2,:)')/2;
Cgauss = PrincipalCurvatures(1,:)'.* PrincipalCurvatures(2,:)';
CsqrtGauss = sign(Cgauss).*sqrt(abs(Cgauss));

if doplot
    figure,patch('vertices',V,'faces',F,'FaceVertexCData',Cmax,'FaceColor','interp','EdgeColor','none','FaceAlpha',1);axis equal, axis off, colorbar,title('Cmax'),addRot3D
    figure,patch('vertices',V,'faces',F,'FaceVertexCData',Cmin,'FaceColor','interp','EdgeColor','none','FaceAlpha',1);axis equal, axis off, colorbar,title('Cmin'),addRot3D
    figure,patch('vertices',V,'faces',F,'FaceVertexCData',Cmean,'FaceColor','interp','EdgeColor','none','FaceAlpha',1);axis equal, axis off, colorbar,title('Cmean'),addRot3D
    figure,patch('vertices',V,'faces',F,'FaceVertexCData',CsqrtGauss,'FaceColor','interp','EdgeColor','none','FaceAlpha',1);axis equal, axis off, colorbar,title('CsqrtGauss'),addRot3D    
end

end