function [T,areas] = getFlatteningDiffCoefMatrix(V,F)
if ispc
 [T,areas]=computeMeshTranformationCoeffsMex(F,V);
 return;
end
V=V';
F=F';
% prepare stuff
nVert = size(V,2);
nFaces = size(F,2);
B = eye(3) - 1/3; % centering matrix

% calculate flattenning coefficients
I = cell(nFaces,1);
J = cell(nFaces,1);
S = cell(nFaces,1);
counter = 1;
allFlatX=nan(length(F),3);
allFlatY=nan(length(F),3);
for ii = 1:nFaces
    % get current triangle
    currF = F(:,ii);
    currV = V(:,currF);
    % trasform to plane
    RFlat = find_2d_embedding(currV);
    currVFlat = RFlat*currV;
    allFlatX(ii,:)=currVFlat(1,:);
    allFlatY(ii,:)=currVFlat(2,:);
    % calculate differential
    currT = B/(currVFlat*B);
    currT(abs(currT)<=1e2*eps) = 0;
    % calculate indices in full tranformation matrix
    I{ii} = [counter counter counter; counter+1 counter+1 counter+1];
    J{ii} = [currF'; currF'];
    S{ii} = currT';
    counter = counter + 2;
end
areas=polyarea(allFlatX',allFlatY')';
% gather
I = cat(1,I{:});
J = cat(1,J{:});
S = cat(1,S{:});
T = sparse(I,J,S,2*nFaces,nVert);



v = randn(size(V,2),2);


% Aold = permute(reshape(T*v,2,[],2),[1 3 2]);
% Anew = reshape(Tnew*v(:),2,2,[]);
% 
% for ii = 1:size(Aold,3)
%     assert(norm(Anew(:,:,ii)'*Anew(:,:,ii)-Aold(:,:,ii)'*Aold(:,:,ii),'fro')/norm(Anew(:,:,ii),'fro')^2<1e-10)
% end

% 
% 

end
