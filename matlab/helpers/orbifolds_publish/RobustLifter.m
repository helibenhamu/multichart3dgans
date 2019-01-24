classdef RobustLifter < handle
    %LIFTER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        tree;%adjacency matrix of the spanning tree of the source mesh
        treeRoot;%the root of the spanning tree
        treeRootOnTarget;%the corresponding index to the treeRoot on the target mesh (boundaries are congruent)
        source;%source mesh (BDMesh)
        target;%target mesh (BDMesh)
        V_t;%the coordinates of the *flattened* target mesh
        V_s;%the coordinates of the *flattened* source mesh
        boundary;%the boundaries of the two meshes
        triangulation_t;%triangulation of the flattened target mesh (for computing bar coords)
        
        %output
        assignedTri;%the indices of the assigned triangles, 1 per source vertex
        barCoords;%the barycentric coordinates of the source vertex according to its corresponding target triangle
        barCoordsMat;%the barycentric coordinates matrix
        curRing;
        curEdge;
        callDepth=0;
        gS;
        T;
        stack={};
        pathEnds;
        vertexCrossDepth=[];
        liftingTime=[];
        liftedSoFar=0;
    end
    
    methods
        function obj=RobustLifter(source,target,gS,T,pathEnds)
            obj.gS=gS;
            obj.T=T;
            obj.pathEnds=pathEnds;
            %make sure can llift
            %             [inj,ind]=source.testLocalBijectivity();
            %             if ~inj
            %                 ind=find(~ind);
            %                 figure()
            %                 patch('vertices',source.TR.Points,'faces',source.TR.ConnectivityList,'facecolor','white','edgecolor','k');
            %                 hold on;
            %                 scatter3(source.TR.Points(ind,1),source.TR.Points(ind,2),source.TR.Points(ind,3),100,'g','filled');
            %                                 axis equal
            %
            %                 error('source is not locally bijective');
            %             end
            %             if ~target.testLocalBijectivity()
            %                 error('target is not locally bijective');
            %             end
            obj.callDepth=0;
            
            obj.source=source;
            obj.target=target;
            obj.V_t=double(target.Y);
            
            obj.V_s=double(source.Y);
            obj.triangulation_t=triangulation(obj.target.TR.ConnectivityList,obj.V_t);
            %set up data for checking for edge intersection on target
            
            obj.assignedTri=zeros(size(obj.V_s,1),1);
            obj.barCoords=nan(size(obj.V_s,1),3);
            obj.vertexCrossDepth=zeros(size(obj.V_s,1),1);
            obj.liftingTime=zeros(size(obj.V_s,1),1);
            obj.barCoordsMat=sparse(size(obj.V_s,1),size(obj.V_t,1));
            
            
            [e,obj.treeRootOnTarget]=obj.findStartEdge();
            obj.treeRoot=e(1);
            obj.tree=mesh_MST(source.TR.ConnectivityList,e,obj.pathEnds(:,1));%setdiff(boundary(:,1))
            %             figure(11);
            %             clf
            %             patch('faces',obj.target.TR.ConnectivityList,'vertices',obj.V_t,'facecolor','none');
            %             hold on
            %             gplot(obj.tree,obj.V_s);
            
        end
        function [edge,root2]=findStartEdge(obj)
            found=0;
            for i=length(obj.pathEnds):-1:1
                
                sp=obj.pathEnds(i,1);
                root2=obj.pathEnds(i,2);
                ringtris=find(any(ismember(obj.target.TR.ConnectivityList,root2),2));
                ring=onering(ringtris,root2,obj.triangulation_t);
                nbrs=unique(obj.source.TR.ConnectivityList(any(ismember(obj.source.TR.ConnectivityList,sp),2),:));
                
                for j=1:length(nbrs)
                    nbr=nbrs(j);
                    if nbr==sp
                        continue;
                    end
                    edge=[sp nbr];
                    edgex=obj.V_s(edge,1);
                    edgey=obj.V_s(edge,2);
                    
                    
                    
                    a=ring.intersectWithBoundary(edgex,edgey);
                    if isempty(a)
                        a=ring.pointInTri([edgex(2) edgey(2)]);
                    end
                    if ~isempty(a)
                        found=1;
                        break;
                    end
                    
                    
                end
                if found
                    return;
                end
                
            end
            error('didn''t find starting edge');
        end
        function lift(obj)
            warning('off','MATLAB:triangulation:PtsNotInTriWarnId');
            %lift a the mesh - generate the bijective mapping
            disp('lifting...');
            %initialize - find one ring of tree root
            
            for i=1:size(obj.pathEnds,1)
                ringtris=find(any(ismember(obj.target.TR.ConnectivityList,obj.pathEnds(i,2)),2));
                tri=ringtris(1);
                bc=double(obj.target.TR.ConnectivityList(tri,:)==obj.pathEnds(i,2));
                obj.setAssignment(obj.pathEnds(i,1),tri,bc,0);
            end
            ringtris=find(any(ismember(obj.target.TR.ConnectivityList,obj.treeRootOnTarget),2));
            ring=onering(ringtris,obj.treeRootOnTarget,obj.triangulation_t);
            
            
            father=obj.V_s(obj.treeRoot,:);
            %iterate over sons of tree root
            sons=find(obj.tree(obj.treeRoot,:));
            for i=1:length(sons)
                obj.addToQueue(father,ring,sons(i),obj.V_s,0);
            end
            obj.assignPointsNew();
            for i=1:size(obj.pathEnds,1)
                ringtris=find(any(ismember(obj.target.TR.ConnectivityList,obj.pathEnds(i,2)),2));
                tri=ringtris(1);
                bc=double(obj.target.TR.ConnectivityList(tri,:)==obj.pathEnds(i,2));
                obj.setAssignment(obj.pathEnds(i,1),tri,bc,0);
            end
            disp('finished!');
            warning('on','MATLAB:triangulation:PtsNotInTriWarnId');
        end
        function setAssignment(obj,source_v_ind,target_tri_ind,barCoords,crossDepth)
            %set the assignment for a given source vertex, to be the in the
            %given target triangle, having the given barycentric coords in
            %it
            obj.assignedTri(source_v_ind)=target_tri_ind;
            obj.barCoords(source_v_ind,:)=barCoords;
            obj.barCoordsMat(source_v_ind,obj.target.TR.ConnectivityList(target_tri_ind,:))=barCoords;
            obj.vertexCrossDepth(source_v_ind)=crossDepth;
            obj.liftedSoFar=obj.liftedSoFar+1;
            obj.liftingTime(source_v_ind)=obj.liftedSoFar;
        end
        function visualizeDebug(obj,fatherring,edgex,edgey)
            figure(5);
            clf
            hold on;
            patch('vertices',obj.V_t,'faces',obj.target.TR.ConnectivityList,'facecolor','none','edgecolor',[0.5,0.5,0.5]);
            
            fatherring.visualize();
            plot(edgex,edgey,'color','red');
            hold off;
        end
        function crossSeam(obj,edge,father,root,Vs,crossDepth)
            fprintf('cross, depth: %d\n',crossDepth);
            %first check what type of 1-ring are we crossing; generally of
            %an edge
            if length(edge)==3%if tri its a bug
                figure();
                patch('vertices',obj.V_t,'faces',obj.target.TR.ConnectivityList,'facecolor','none','edgecolor',[0.9,0.9,0.9]);
                hold on;
                patch('vertices',obj.V_t,'faces',edge,'facecolor','green','linewidth',2,'edgecolor','green');
                edgex=[father(1) obj.V_s(root,1)];
                edgey=[father(2) obj.V_s(root,2)];
                plot(edgex,edgey,'color','red');
                error('cannot cross seam if in triangle, root: %d, edge: (%d,%d)',root,edge(1),edge(2));
            end
            ring=[];
            if length(edge)==1%if 1ring was a vertex
                
                %some warnings and debugging visualization
                warning('crossing edge from vertex');
                figure(21);
                clf
                patch('vertices',obj.V_t,'faces',obj.target.TR.ConnectivityList,'facecolor','none');
                hold on;
                edgex=[father(1) Vs(root,1)];
                edgey=[father(2) Vs(root,2)];
                plot(edgex,edgey,'green','linewidth',4);
                 pause;
                 
                 
                %first
               
                found=false;
                for i=1:length(obj.target.pathPairs)
                    for j=1:2
                        inds=find(ismember(obj.target.pathPairs{i}(:,j),edge));
                        if ~isempty(inds)
                            found=true;
                            break;
                        end
                    end
                    if found
                        break;
                    end
                end
                assert(~isempty(inds));
                if inds<length(obj.target.pathPairs{i})
                    edge=[edge obj.target.pathPairs{i}(inds+1,j)];
                else
                    edge=[obj.target.pathPairs{i}(inds-1,j) edge ];
                end
                %                 angle = atan2(norm(cross(a,b)),dot(a,b));
                 v_ind=obj.target.pathPairs{i}(inds,3-j);
                ringtris=find(any(ismember(obj.target.TR.ConnectivityList,v_ind),2));
                
                
                ring=onering(ringtris,v_ind,obj.triangulation_t);
                
            end
            found=0;
            for i=1:length(obj.target.pathPairs)
                for j=1:2
                    inds=find(ismember(obj.target.pathPairs{i}(:,j),edge));
                    
                    if length(inds)==2
                        found=true;
                        break;
                    end
                end
                if found
                    break;
                end
            end
            if ~found
                figure(5);
                clf
                hold on;
                patch('vertices',obj.V_t,'faces',obj.target.TR.ConnectivityList,'facecolor','none','edgecolor',[0.5,0.5,0.5]);
               
                cross_edgex=obj.V_t(edge,1);
                cross_edgey=obj.V_t(edge,2);
                edgex_old=[father(1) Vs(root,1)];
                edgey_old=[father(2) Vs(root,2)];
            %                                 edgey_old=[father(2) Vs(root,2)];
                                            plot(cross_edgex,cross_edgey,'color','green','linewidth',3);
                                            plot(edgex_old,edgey_old,'color','red','linewidth',3);
                error('reached outside!');
            end
            assert(length(inds)==2);
            R=double(obj.gS{i});
            T=double(obj.T{i});
            if j==2
                %R*x1+T=x2e
                %x1=R^-1*x2-R^-1*T
                R=inv(R);
                T=-T*R';
            end
            if obj.target.pathPairs{i}(inds(1),j)~=edge(1)
                inds=inds(2:-1:1);
            end
            
            assert(all(obj.target.pathPairs{i}(inds,j)==edge(:)));
            int_e=obj.target.pathPairs{i}(inds,3-j);
            
            err=obj.V_t(int_e,:)-obj.V_t(edge,:)*R'-[T;T];
%             assert(max(abs(err(:)))<1e-4);
            if isempty(ring)
                ringtris=find(sum(ismember(obj.target.TR.ConnectivityList,int_e),2)==2);
                %                 assert(length(ringtris)==2);
                ring=onering(ringtris,int_e,obj.triangulation_t);
            end
            Vs_new=Vs*R'+repmat(T,length(Vs),1);
            father_new=father*R'+T;
            %             clf
            %                              figure(5);
            %                                 patch('vertices',obj.V_t,'faces',obj.target.TR.ConnectivityList,'facecolor','none');
            %                                 hold on;
            %                                 edgex_old=[father(1) Vs(root,1)];
            %                                 edgey_old=[father(2) Vs(root,2)];
            %                                 edgex_new=[father_new(1) Vs_new(root,1)];
            %                                 edgey_new=[father_new(2) Vs_new(root,2)];
            %                                 plot(edgex_old,edgey_old,'color','green','linewidth',3);
            %                                 plot(edgex_new,edgey_new,'color','red','linewidth',3);
            %                                 oldpath=obj.target.pathPairs{i}(:,j);
            %                                 newpath=obj.target.pathPairs{i}(:,3-j);
            %                                 plot(obj.V_t(oldpath,1),obj.V_t(oldpath,2),'color',[0 1 1]);
            %                                 plot(obj.V_t(newpath,1),obj.V_t(newpath,2),'color',[1 0 1]);
            %                                 axis equal
            %                                 pause
            
            obj.addToQueue(father_new,ring,root,Vs_new,crossDepth+1);
        end
        function addToQueue(obj,father,fatherring,root,Vs,crossDepth)
            a=[];
            a.father=father;
            a.fatherring=fatherring;
            a.Vs=Vs;
            a.root=root;
            a.crossDepth=crossDepth;
            obj.stack{end+1}=a;
            
        end
        function [father,fatherring,root,Vs,crossDepth]=popFromQueue(obj)
            if isempty(obj.stack)
                father=[];
                fatherring=[];
                root=[];
                Vs=[];
                crossDepth=[];
                return;
            end
            a=obj.stack{end};
            obj.stack(end)=[];
            father=a.father;
            fatherring=a.fatherring;
            Vs=a.Vs;
            root=a.root;
            crossDepth=a.crossDepth;
        end
        function assignPointsNew(obj)
            while(true)
                [father,fatherring,root,Vs,crossDepth]=obj.popFromQueue();
                if isempty(father)
                    break;
                end
                obj.callDepth=obj.callDepth+1;
                obj.curRing=fatherring;
                obj.curEdge=[];
                obj.curEdge.father=father;
                obj.curEdge.root=root;
                % find intersection point
                edgex = [father(1),Vs(root,1)];
                edgey = [father(2),Vs(root,2)];
                
                %             if obj.callDepth>400
                if mod(obj.callDepth,1000)==0
                    %                     disp('draw');
                    figure(5);
                    obj.callDepth
                    obj.visualizeDebug(fatherring, edgex,edgey);
                    pause(0.1);
                end
                %                 disp('click to continue.....');
                %                 pause;
                %             end
                %             figure(2);
                %             clf
                %             plot(edgex,edgey,'color','red');
                %             hold on;
                %             fatherring.visualize();
                %             pause(0.01);
                [int_p,int_e]=fatherring.intersectWithBoundary(edgex',edgey');
                if var(edgex)+var(edgey)<1e-10
                    
                    %set root to be on father
                    Vs(root,:)=father;
                    %father lies on generator of 1-ring find any tri that
                    %touches it
                    tri_ind=find(sum(ismember(obj.target.TR.ConnectivityList,fatherring.generator),2)==length(fatherring.generator));
                    assert(~isempty(tri_ind));
                    tri_ind=tri_ind(1);
                    bc=cartesianToBarycentric(obj.triangulation_t,tri_ind,Vs(root,:));
                    obj.setAssignment(root,tri_ind,bc,crossDepth);
                    sons=find(obj.tree(root,:));
                    for i=1:length(sons)
                        obj.addToQueue(Vs(root,:),fatherring,sons(i),Vs,crossDepth);
                    end
                    continue;
                end
                %check if not intersecting anythin
                if isempty(int_e)
                    
                    %check if root is inside a tri of the target 1ring
                    tri_ind=fatherring.pointInTri(Vs(root,:));
                    
                    if isempty(tri_ind)
                        % OUTSIDE CASE
                        %                     obj.visualizeDebug(fatherring, edgex,edgey);
                        %                     error('reached outside');
                        e=fatherring.generator;
                        
                        obj.crossSeam(e,father,root,Vs,crossDepth);
                        continue;
                    else
                        %                     disp('====== in tri ========');
                        bc=cartesianToBarycentric(obj.triangulation_t,tri_ind,Vs(root,:));
                        obj.setAssignment(root,tri_ind,bc,crossDepth);
                        %                     ringtris=find(any(ismember(obj.target.TR.ConnectivityList,obj.target.TR.ConnectivityList(tri_ind,:)),2));
                        ringtris=tri_ind;
                        if all(obj.target.TR.ConnectivityList(tri_ind,:)== [636         590        4830])
                            disp('a')
                        end
                        ring=onering(ringtris,obj.target.TR.ConnectivityList(tri_ind,:),obj.triangulation_t);
                        sons=find(obj.tree(root,:));
                        for i=1:length(sons)
                            obj.addToQueue(Vs(root,:),ring,sons(i),Vs,crossDepth);
                        end
                        continue;
                    end
                elseif length(int_e)==1%%vertex case
                    %                 disp('====== in vertex ========');
                    v_ind=int_e;
                    ringtris=find(any(ismember(obj.target.TR.ConnectivityList,v_ind),2));
                    tri=ringtris(1);
                    bc=double(obj.target.TR.ConnectivityList(tri,:)==v_ind);
                    obj.setAssignment(root,tri,bc,crossDepth);
                    sons=find(obj.tree(root,:));
                    ring=onering(ringtris,v_ind,obj.triangulation_t);
                    for i=1:length(sons)
                        obj.addToQueue(obj.V_t(v_ind,:),ring,sons(i),Vs,crossDepth);
                    end
                    continue;
                    
                    
                elseif length(int_e)==2 %EDGE CASE
                    %                 disp('====== in edge ========');
                    ringtris=find(sum(ismember(obj.target.TR.ConnectivityList,int_e),2)==2);
                    %                 assert(length(ringtris)==2);
                    ring=onering(ringtris,int_e,obj.triangulation_t);
                    obj.addToQueue(int_p,ring,root,Vs,crossDepth);
                else
                    error('int_e''s length should be 0, 1 or 2');
                end
                
            end
        end
        
        
    end
    
end

