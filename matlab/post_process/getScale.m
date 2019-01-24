function [ scale ] = getScale( method, varargin)
%getScale returns the scale
%       method -
% -------------------------------------------------------------------------
%           'original' - returns the conformal distortion per vertex,
%                        averaged over the adjacent faces according to
%                        their area from template flattening
%               Inputs:
%                        fInfo - fInfo structs that contain the flattening
%                                object
% -------------------------------------------------------------------------
%           'grid'     - returns the conformal distortion per vertex
%                        by calculating the distortion per triangle in the 
%                        regular grid of the image and then interpolating.
%               Inputs:
%                       fInfo - fInfo structs that contain the flattening
%                               object
%                       data  - grid to lift scale from
% -------------------------------------------------------------------------
%           'lift'     - returns the conformal distortion per vertex
%                        averaged over the adjacent faces according to
%                        their area from lift of charts



switch method
    case 'original'
        if nargin<2
            error('Inputs are missing');
        end
        fInfo = varargin{1};
        flattener = {};
        for jj=1:length(fInfo)
            flattener{jj} = fInfo{1,jj}.flattener;
            scale_cut = flattener{jj}.vertexScale();
            scale(:,jj) = flattener{jj}.valsOnUncutMesh(scale_cut);
        end
        scale(scale<0) = min(colStack(abs(scale)));
    
    case 'grid'        
        if nargin<3
            error('Inputs are missing');
        end
        fInfo = varargin{1};
        data = varargin{2};
        
        % calculate scale distortion from triangles on grid to 3D
        
        % extract 3D coordinates of triangle vertices
        sz = size(data,1);        
        V = reshape(data, [sz^2, size(data,3)]);
        V = reshape(V, [size(V,1), 3, length(fInfo)]);  
        % triangulate
        [X, Y] = meshgrid(linspace(-3,1,sz),linspace(-3,1,sz));
        faces = delaunay(X(:),Y(:));
        TRI = triangulation(faces,[X(:),Y(:)]);
        
        % calculate areas
        for ii = 1:length(fInfo)
           [~ , areas{ii}] =  CORR_calculate_area(faces,squeeze(V(:,:,ii)));            
        end
        areas = cell2mat(areas);
        areas(areas==0) = nan;
        scale_grids = 1./areas;
        scale_grids(isnan(scale_grids)) = 0;
        
        for jj = 1:length(fInfo)
            flattener = {};
            flattener{jj} = fInfo{1,jj}.flattener;
            flat_V = flattener{jj}.flat_V;
            
            V1 = flat_V(:,1);
            V1(V1>1) = V1(V1>1)-4;
            V1(V1<-3) = V1(V1<-3)+4;
            V2 = flat_V(:,2);
            V2(V2<-3) = V2(V2<-3)+4;
            V2(V2>1) = V2(V2>1)-4;
            
            faces_idx = pointLocation(TRI,[V1,V2]);           
            scale(:,jj) = flattener{jj}.valsOnUncutMesh(scale_grids(faces_idx,jj));

        end
        scale = scale';
       
    case 'lift'
        if nargin<4
            error('Inputs are missing');
        end
        invPsi = varargin{1};
        phi = varargin{2};
        params = varargin{3};
        chartFlags = [0 , cumsum(params.numV)*2];
        for ii = 1:params.num_charts           
            cutV = squeeze(invPsi(ii,params.TempMeshes{ii}.cutIndsToUncutInds,:));
            [V2A,areas] = getFlatteningDiffCoefMatrix(cutV,params.TempMeshes{ii}.T); % calculate map between 2d vertices to differentials
            V_flat = phi((1+chartFlags(ii)):chartFlags(ii+1));
            if ispc
                As = reshape((V2A*V_flat)',2,2,[]);
            else
                As = reshape((V2A*reshape(V_flat,params.numV(ii),2))',2,2,[]);
            end
            a = squeeze(As(1,1,:))';
            b = squeeze(As(1,2,:))';
            c = squeeze(As(2,1,:))';
            d = squeeze(As(2,2,:))';%the entries of A
            
            dets=(a.*d-b.*c)';
            
            % per face - area* det
            c=areas.*dets;
            %accumulate c per vertex
            vals=repmat(c,1,3);
            v=accumarray(params.TempMeshes{ii}.cutIndsToUncutInds(params.TempMeshes{ii}.T(:))',vals(:));
            %accumulate areas per vertex 
            areas=repmat(areas,1,3);
            vareas=accumarray(params.TempMeshes{ii}.cutIndsToUncutInds(params.TempMeshes{ii}.T(:))',areas(:));
            %average the scale by areas
            vc=v./vareas;     
            scale(ii,:)=vc(cellfun(@(X)X(1),params.TempMeshes{ii}.uncutIndsToCutInds));
        end          
        
end

