function [Cmin,Cmax,Cmean,Cgauss] = calcCurvature(V,F,params)
% V,F mesh representation

% normalize shape by shape area
area = CORR_calculate_area(F,V);
V = V/sqrt(area);
[Cmax, Cmin,Cmean, Cgauss] = getCurvatureFeatures(V,F);

end