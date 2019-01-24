classdef Flattener < handle
    %FLATTENER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        M_orig; %struct of original mesh
        inds; %indices of the cones
        singularities; %cone angles
        flat_V; %will hold flat vertices after solcing
        flat_T; %the triangles of the mesh after cutting (if needed)
        M_cut; %the cut mesh
        flipped; %boolean - true for flipped (det<0) tris
        V2A; %matrix 
        As;
        dets;
        frobenius;
        smin;
        smax;
        areas;
        cut_colors={[0 0.4 1],[0 0.7 0.3],[ 0.8 0.2 0.0],[0.8,0.6,0]};
        LM_colors={[0.6 0.2 0.1],[1 0.8 0],[1 0 1],[0 0.8 0.4]};
        orgBoundary=[];
        isDisc=false;
        verbose=true;
    end
    
    methods
        function obj=Flattener(M_orig,inds,singularities,M_cut)
            %             assert(length(inds)==length(singularities)+1);
            obj.M_orig=M_orig;
            obj.inds=inds;
            obj.singularities=singularities;
            if nargin>3
                obj.setCutMesh(M_cut);
            end
            TR=triangulation(obj.M_orig.F',obj.M_orig.V');
            t=(TR.freeBoundary());
            if ~isempty(t)
                obj.orgBoundary=t(:,1);
%                 if ~all(ismember(inds,obj.orgBoundary))
%                     error('in case of a disk orbifold, all cones should be boundary vertices!');
%                 end
%                 ind=find(obj.inds(1)==obj.orgBoundary);
%                 obj.orgBoundary=[obj.orgBoundary(ind:end)' obj.orgBoundary(1:ind-1)']';
            end
%             obj.isDisc=~isempty(obj.orgBoundary);
            
        end
        function vals_on_orig_mesh=valsOnUncutMesh(obj,vals_on_cut_mesh)
            
                        vals_on_orig_mesh=vals_on_cut_mesh(cellfun(@(X)X(1),obj.M_cut.uncutIndsToCutInds));
            
        end
        function vc=vertexScale(obj)
            %return the conformal distortion per vertex, averaged over the
            %adjacent faces according to their area
            %computing the distortion metrics per face
            obj.computeDistortion();
            % per face - area* det
            c=obj.areas.*obj.dets;
            %accumulate c per vertex
            vals=repmat(c,1,3);
            %v=accumarray(obj.flat_T(:),vals(:));
            v=accumarray(obj.M_cut.cutIndsToUncutInds(obj.flat_T(:))',vals(:));
            %accumulate areas per vertex 
            areas=repmat(obj.areas,1,3);
            %vareas=accumarray(obj.flat_T(:),areas(:));
            vareas=accumarray(obj.M_cut.cutIndsToUncutInds(obj.flat_T(:))',areas(:));
            %average the scale by areas
            vc=v./vareas;
            vc = vc(obj.M_cut.cutIndsToUncutInds); 

        end
        function computeDistortion(obj,force)
            if nargin<2
                force=false;
            end
            if ~force && ~isempty(obj.dets)
                
                return;
            end
            if isempty(obj.As)||force
                obj.computeAs();
            end
            fprintf('==== Computing distortion etc. ===\n');
            tid=tic;
            As=obj.As;
            
            
            a = squeeze(As(1,1,:))';
            b = squeeze(As(1,2,:))';
            c = squeeze(As(2,1,:))';
            d = squeeze(As(2,2,:))';%the entries of A
            
            obj.dets=(a.*d-b.*c)';
            obj.frobenius=sqrt(a.^2+b.^2+c.^2+d.^2)';
            
            
            alpha = [a+d;b-c]/2; %2XM
            beta = [a-d;b+c]/2;
            alpha=sqrt(sum(alpha.^2));
            beta=sqrt(sum(beta.^2));
            obj.smax=(alpha+beta)';
            obj.smin=abs((alpha-beta)');
            
            obj.flipped=obj.dets<-1e-5;
            if(any(obj.flipped))
                
                warning('*** THERE ARE FLIPPED TRIS!!! ***');
                fprintf('area of *largest* flipped tri: %f\n',max(abs(obj.dets(obj.flipped))));
            end
        end

        function [orgVals,cutVals]=liftImage(obj,IM,classic,method)
            % lifting the image (grid) back to the original mesh
            % inputs:
            %       IM - toric image (4 rotated copies)
            %       classic - ask Haggai
            %       method - default is 'interp2'
            %           two methods are available:
            %            1. 'nn' - nearest neighbor (written by Noam)
            %            2. 'interp2' - interpolating over the grid
            
            if nargin<4
                method = 'interp2';
                if nargin<3
                    classic=false;
                end
            end
            assert(size(IM,1)==size(IM,2),'image should be square!');
            sz = size(IM,1);
            
            if strcmp(method,'nn')
                IM = IM(floor(size(IM,1)/2+1:end),floor(size(IM,2)/2)+1:end);
                if ~classic
                t=Tiler(obj.flat_V,obj.flat_T,obj.M_cut.pathPairs);
                t.tile(6);
                orgV=obj.flat_V;


                found=false(length(orgV),1);
                V=obj.flat_V;
                for i=1:length(t.trans)
                    A=t.trans{i};
                    curV=orgV*A([1 2],:)'+repmat(A(3,:),length(obj.flat_V),1);

                    good=all(curV>=-1&curV<=1,2)&~found;
                    found(good)=true;
                    V(good,:)=curV(good,:);
                    if isempty(good)
                        break;
                    end
                end
                end
                %move from [-1,1]^2 to [0,1]^2
                V=(V+1)/2;
                % we want 0 mapped to 1, and 1 mapped to length(image)
                V=V*(length(IM)-1);
                V=V+1;
                V(V<1)=1;
                V(V>length(IM))=length(IM);
                inds_from_pos=round(V);
                I=sub2ind(size(IM),inds_from_pos(:,1),inds_from_pos(:,2));
                cutVals=IM(I);
                orgVals=cutVals(cellfun(@(X)X(1),obj.M_cut.uncutIndsToCutInds));
            elseif strcmp(method,'interp2')
                V=obj.flat_V;
                V1 = V(:,1);
                V1(V1>1) = V1(V1>1)-4;
                V1(V1<-3) = V1(V1<-3)+4;
                V2 = V(:,2);
                V2(V2<-3) = V2(V2<-3)+4;
                V2(V2>1) = V2(V2>1)-4;
                
                [X,Y] = meshgrid(linspace(-3,1,sz),linspace(-3,1,sz));
                cutVals = interp2(X,Y,IM,V1,V2);
                orgVals=cutVals(cellfun(@(X)X(1),obj.M_cut.uncutIndsToCutInds));        
                
            end
                
        end
        
        
        
        
        function B=computeAffineMinimizer(obj,opt)
            obj.computeAs();
            
            As = obj.V2A*obj.flat_V(:);
            a=As(1:4:end);
            b=As(3:4:end);
            c=As(2:4:end);
            d=As(4:4:end);
            correctAs=zeros(size(As));
            correctAs([1:4:end 2:4:end 3:4:end 4:4:end])=[a b c d];
            B=minimize_global_lscm(correctAs,obj.areas);
        end
        function correctGlobalAffine(obj,opt)
            if nargin==1
                opt='LSCM';
            end
            B=obj.computeAffineMinimizer(opt);
            obj.flat_V=obj.flat_V*B';
            obj.fixToAxis();
            obj.computeDistortion(true);
        end
        function setCutMesh(obj,M_cut)
            M_cut.V=obj.M_orig.V(:,M_cut.cutIndsToUncutInds);
            
            if size(M_cut.V,2)~=3
                M_cut.V=M_cut.V';
            end
            if size(M_cut.T,2)~=3
                M_cut.T=M_cut.T';
            end
            obj.M_cut=CutMesh(M_cut.V,M_cut.T,M_cut.pathPairs,M_cut.cutIndsToUncutInds,M_cut.uncutIndsToCutInds);
        end
        
        function cut(obj)
            if obj.verbose
                fprintf('*** cutting: ');
                tid=tic;
            end
            if length(obj.inds)==3
                root=length(obj.inds);
                fixedPairs=[ones(1,length(obj.inds)-1)*root;1:length(obj.inds)-1]';
            else
                root=1;
                fixedPairs=[1 3;3 4;4 2];
            end
            tree=sparse(fixedPairs(:,1),fixedPairs(:,2),1,length(obj.inds),length(obj.inds));
            cutter=TreeCutter(obj.M_orig.V',obj.M_orig.F',tree,obj.inds,root);
            cutter.cutTree();
            obj.setCutMesh(cutter);
            if obj.verbose
                toc(tid)
            end
        end
        
        function setLinearConstraints(obj,cons,v1,v2,inds)
            d=sqrt(sum(obj.M_cut.V(inds(1:end-1),:)-obj.M_cut.V(inds(2:end),:),2).^2);
            d=[0;cumsum(d)/sum(d)];
            for i=1:length(inds)-1
                cons.addConstraint(inds(i),1,v1*(1-d(i))+v2*d(i));
            end
        end
        function flatten(obj,convexBoundary,DIRICHLET)
            %convexBoundary - 1. omitted or false for free boundary
            %2. 'square' for arrangment on a square
            %3. true for disc
            if nargin<2
                convexBoundary=false;
            end
            if nargin<3
                DIRICHLET=true;
            end
           
            if isempty(obj.M_cut)
                if ~obj.isDisc
                    
                    
                    obj.cut();
                else
                    if strcmp(convexBoundary,'square') || strcmp(convexBoundary,'freesquare')
                        obj.orgBoundaryToPaths(4);
                    elseif strcmp(convexBoundary,'freetri')
                        obj.orgBoundaryToPaths(3);
                    else
                        error;
                    end
                end
            end
            if obj.verbose
                fprintf('*** flattening: ');
            end
            startP=obj.M_cut.uncutIndsToCutInds{obj.inds(1)};
            assert(length(startP)==1);
            
            tid=tic;
            TR=triangulation(obj.M_cut.T,obj.M_cut.V);
            
            
            
            
            cons=PosConstraints(length(obj.M_cut.V));
            
            
            if convexBoundary
                if strcmp(convexBoundary,'square')
                    pathEnds=[];
                    for i=1:length(obj.M_cut.pathPairs)
                        pathEnds=[pathEnds obj.M_cut.pathPairs{i}([1 end],:)];
                    end
                    pathEnds=unique(pathEnds);
                    all_binds = TR.freeBoundary();
                    all_binds(any(ismember(all_binds,obj.orgBoundary),2),:)=[];
                    assert(all(all_binds(:,2)==all_binds([2:end,1],1)));
                    all_binds=all_binds(:,1);
                    all_binds=all_binds(end:-1:1);
                    ind=find(all_binds==startP);
                    all_binds=all_binds([ind:end,1:ind-1]);
                    p=find(ismember(all_binds,pathEnds));
                    obj.setLinearConstraints(cons,[-1 1]',[1 1]',all_binds(p(1):p(2)));
                    obj.setLinearConstraints(cons,[1 1]',[1 -1]',all_binds(p(2):p(3)));
                    obj.setLinearConstraints(cons,[1 -1]',[-1 -1]',all_binds(p(3):p(4)));
                    obj.setLinearConstraints(cons,[-1 -1]',[-1 1]',[all_binds(p(4):length(all_binds)); all_binds(1)]);
                elseif strcmp(convexBoundary,'freesquare')
                    %                     if obj.isDisc
                    %                         obj.orgBoundaryToPaths(4);
                    %                     end
                    pathEnds=[];
                    for i=1:length(obj.M_cut.pathPairs)
                        pathEnds=[pathEnds obj.M_cut.pathPairs{i}([1 end],:)];
                    end
                    pathEnds=unique(pathEnds);
                    all_binds = TR.freeBoundary();
                     all_binds(any(ismember(all_binds,obj.orgBoundary),2),:)=[];
                    all_binds=all_binds(end:-1:1,1);
                    ind=find(all_binds==startP);
                    all_binds=all_binds([ind:end,1:ind-1]);
                    p=find(ismember(all_binds,pathEnds));
                    %                     p=all_binds(p);
                    cons.addConstraint(all_binds(p(1)),1,[-1 1]');
                    cons.addConstraint(all_binds(p(2)),1,[1 1]');
                    cons.addConstraint(all_binds(p(3)),1,[1 -1]');
                    cons.addConstraint(all_binds(p(4)),1,[-1 -1]');
                    for i=p(1)+1:p(2)-1
                        cons.addLineConstraint(all_binds(i),[0 1],1);
                    end
                    
                    for i=p(2)+1:p(3)-1
                        cons.addLineConstraint(all_binds(i),[1 0],1);
                    end
                    
                    
                    for i=p(3)+1:p(4)-1
                        cons.addLineConstraint(all_binds(i),[0 -1],1);
                    end
                    for i=p(4)+1:length(all_binds)
                        cons.addLineConstraint(all_binds(i),[-1 0],1);
                    end
                    
                    
                elseif strcmp(convexBoundary,'freetri')
                    %                     if obj.isDisc
                    %                         obj.orgBoundaryToPaths(4);
                    %                     end
                    pathEnds=[];
                    for i=1:length(obj.M_cut.pathPairs)
                        pathEnds=[pathEnds obj.M_cut.pathPairs{i}([1 end],:)];
                    end
                    pathEnds=unique(pathEnds);
                    all_binds = TR.freeBoundary();
                     all_binds(any(ismember(all_binds,obj.orgBoundary),2),:)=[];
                    all_binds=all_binds(end:-1:1,1);
                    ind=find(all_binds==startP);
                    all_binds=all_binds([ind:end,1:ind-1]);
                    p=find(ismember(all_binds,pathEnds));
                    %                     p=all_binds(p);
                    cons.addConstraint(all_binds(p(1)),1,[0 0]'); %90
                    cons.addConstraint(all_binds(p(2)),1,[0 1]'); %45
                    cons.addConstraint(all_binds(p(3)),1,[1 0]');  %45
                    
                    for i=p(1)+1:p(2)-1
                        cons.addLineConstraint(all_binds(i),[1 0],0);
                    end
                    
                    for i=p(2)+1:p(3)-1
                        cons.addLineConstraint(all_binds(i),normr([1 1]),sqrt(2)/2);
                    end
                    
                    
                    
                    for i=p(3)+1:length(all_binds)
                        cons.addLineConstraint(all_binds(i),[0 1],0);
                    end
                    
                else
                    
                    all_binds = TR.freeBoundary();
                     all_binds(any(ismember(all_binds,obj.orgBoundary),2),:)=[];
                    d=sqrt(sum(obj.M_cut.V(all_binds(:,1),:)-obj.M_cut.V(all_binds(:,2),:),2).^2);
                    %want area of 4 (as the square), so pi r^2=4 -->
                    %r=2/sqrt(pi)
                    R=2/sqrt(pi);
                    theta=2*pi*cumsum(d)/sum(d);
                    for i=1:length(all_binds)
                        ind=all_binds(i,1);
                        t=theta(i);
                        cons.addConstraint(ind,1,R*[cos(t) sin(t)]');
                    end
                end
            else
                pathEnds=[];
                for i=1:length(obj.M_cut.pathPairs)
                    pathEnds=[pathEnds obj.M_cut.pathPairs{i}([1 end],:)];
                end
                pathEnds=unique(pathEnds);
                all_binds = TR.freeBoundary();
                 all_binds(any(ismember(all_binds,obj.orgBoundary),2),:)=[];
                
                assert(all(all_binds(:,2)==all_binds([2:end,1],1)),'the boundary is not a simple closed loop!');
                all_binds=all_binds(:,1);
                ind=find(all_binds==startP);
                all_binds=all_binds([ind:end,1:ind-1]);
                p=find(ismember(all_binds,pathEnds));
                p=all_binds(p);
                
                %                 P={};
                angs={};
                theta=2*pi*(1:length(p))/length(p)+pi/4;
                coords=[cos(theta)' sin(theta)']*sqrt(2);
                if length(obj.inds)==4
                    %                     tcoords=[0 -0.5;1 -0.5;1 0;0 0.5;1 1];
                    tcoords=[0 -0.5;0 0.5];
                elseif all(obj.singularities==[4 4])
                    tcoords=[-1 -1;1 1];
                else
                    tcoords=coords;
                end
                %                 tcoords=[0 -0.5;0 0.5];warning('ifproblemremovethis');
                for i=1:length(p)
                    ind=find(obj.M_cut.cutIndsToUncutInds(p(i))==obj.inds);
                    if ind<=2
                        cons.addConstraint(p(i),1,tcoords(ind,:)');
                        
                        angs{p(i)}=obj.singularities(ind);
                    else
                        angs{p(i)}=[];
                    end
                    if length(obj.inds)==4&& i==2
                        cons.addConstraint(p(i),1,[1 -0.5]');
                    end
                end
                for i=1:length(obj.M_cut.pathPairs)
                    path1=obj.M_cut.pathPairs{i}(:,1);
                    path2=obj.M_cut.pathPairs{i}(:,2);
                    sign=-1;
                    if path1(end)==path2(end)
                        path1=path1(end:-1:1);
                        path2=path2(end:-1:1);
                        sign=1;
                    end
                    
                    ang=angs{path1(1)};
                    if isempty(ang)
                        ang=1;
                    end
                    
                    if ~isempty(ang)
                        ang=sign*ang;
                        R=[cos(2*pi/ang) -sin(2*pi/ang);sin(2*pi/ang) cos(2*pi/ang)];
                        cons.addTransConstraints(path1,path2,R)
                    end
                end
            end
            if obj.verbose
                fprintf('constraint generation: %f seconds\n',toc(tid));
                tid=tic;
            end
            
            
            %% Dirichlet Laplacian...
            if DIRICHLET
                L=cotmatrix(obj.M_cut.V,obj.M_cut.T);
                
                m=min(min(tril(L,-1)));
                if m<0
                    warning('Mesh is not Delaunay!! fixing...');
                    L(L<0)=1e-2;
                    inds=sub2ind(size(L),1:length(L),1:length(L));
                    L(inds)=0;
                    L(inds)=-sum(L);
                end
            else
                L=mean_value_laplacian(obj.M_cut.V,obj.M_cut.T);
            end
            
            RealL=sparse(size(L,1)*2,size(L,2)*2);
            RealL(1:2:end,1:2:end)=L;
            RealL(2:2:end,2:2:end)=L;
            L=RealL;
            if obj.verbose
                fprintf('compute: %f seconds\n',toc(tid));
                
                tidc=tic;
            end
            x=computeFlattening(cons.A,cons.b,L);
            if obj.verbose
                fprintf('lin solve: %f seconds\n',toc(tidc));
            end
            X=x(1:2:end);
            Y=x(2:2:end);
            
            obj.flat_V=[X Y];
            
            %%
            
            %
            obj.flat_T=obj.M_cut.T;
            
            
            
        end
        function orgBoundaryToPaths(obj,N)
            pathPairs={};
%             inds=obj.orgBoundary;
%             d=sqrt(sum(obj.M_orig.V(:,inds(1:end-1))'-obj.M_orig.V(:,inds(2:end))',2).^2);
%             d=[0;cumsum(d)/sum(d)];
%             
%             inds=[1];
%             for i=1:N-1
%                 ind=find(d>i/N,1);
%                 inds(end+1)=ind;
%             end
            %             inds(end+1)=length(obj.orgBoundary);
            
            for i=1:length(obj.inds)-1
                s=find(obj.orgBoundary==obj.inds(i));
                e=find(obj.orgBoundary==obj.inds(i+1));
                p=obj.orgBoundary(s:e);
                pathPairs{i}=[p p];
            end
             s=find(obj.orgBoundary==obj.inds(end));
            p=obj.orgBoundary([s:length(obj.orgBoundary) 1]);
            pathPairs{end+1}=[p p];
%             obj.inds=obj.orgBoundary(inds);
            
            obj.M_cut=CutMesh(obj.M_orig.V',obj.M_orig.F',pathPairs,1:length(obj.M_orig.V),num2cell(1:length(obj.M_orig.V)));
        end
        
        function computeV2A(obj)
            
            
            [obj.V2A,obj.areas] = getFlatteningDiffCoefMatrix(obj.M_cut.V,obj.M_cut.T); % calculate map between 2d vertices to differentials
            
        end
        
        
        
        function computeAs(obj)
            if isempty(obj.V2A)
                obj.computeV2A();
            end
            
            %obj.As = permute(reshape(obj.V2A*obj.flat_V,2,[],2),[1 3 2]);
            if ispc
                obj.As = reshape((obj.V2A*obj.flat_V(:))',2,2,[]);
            else
            obj.As = reshape((obj.V2A*obj.flat_V)',2,2,[]);
            end
            %A*v=[x y]; --> A=[x y]v^-1
            
        end
        function fixToAxis(obj)
            p=obj.flat_V(obj.M_cut.uncutIndsToCutInds{obj.inds(3)},:);
            p1=p(1,:);
            p2=p(2,:);
            [ M,t ] = computeSimilarity( p1,p2,[1 0],[-1 0] );
            obj.flat_V=bsxfun(@plus,obj.flat_V*M',t');
        end
        function fixToGrid(obj,scale)
            obj.fixToAxis();
            obj.flat_V=obj.flat_V*scale;
            ind1=obj.M_cut.uncutIndsToCutInds{obj.inds(3)}(1);
            ind2=obj.M_cut.uncutIndsToCutInds{obj.inds(3)}(2);
            
            X=obj.flat_V;
            if X(ind1,1)>X(ind2,1)
                ind3=ind1;
                ind1=ind2;
                ind2=ind3;
            end
            X(:,1)=X(:,1)-X(ind1,1);
            m=X(ind2,1);
            X=2*round(m)*X/m;
            
            
            ind3=obj.M_cut.uncutIndsToCutInds{obj.inds(4)}(1);
            ind4=obj.M_cut.uncutIndsToCutInds{obj.inds(4)}(2);
            p1=X(ind3,:);
            p2=X(ind4,:);
            p3=X(ind2,:);
            q1=round(p1);
            q2=round(p2);
            q3=p3;
            [ A,t ] = affineTransFrom3Points( p1,p2,p3,q1,q2,q3 );
            X=bsxfun(@plus,X*A',t');
            obj.flat_V=X;
            
        end
    end
    
end


