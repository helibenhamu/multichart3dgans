classdef TreeCutter < handle
    
    properties
        % mesh variables - initially the original mesh's, and get updated
        % during the cutting.
        V; %the vertices of the mesh
        T; %the triangles
        pathPairs; %seams.
        cutIndsToUncutInds;
        uncutIndsToCutInds;
        
        %cutting variables - all data related to the required cutting
        
        treeStructure; %the adjacencies of the tree to cut according to
        treeIndices; %the indices in the mesh related to the tree
        treeRoot;%the index that is the root of the tree
        
        %flag to make sure we only perform cutting ONCE
        alreadyCut;
        %verbose flag
        VERBOSE=0;
        %tolerance of snapping to vertex
        finishedPaths=0;
        
    end
    
    methods
        function obj=TreeCutter(V,T,tree,treeindices,root)
            assert(all(diag(tree)==0));
            
            obj.V=V;
            obj.T=T;
            obj.pathPairs=[];
            obj.uncutIndsToCutInds=num2cell(1:length(V));
            obj.cutIndsToUncutInds=1:length(V);
            obj.alreadyCut=0;
            if nargin<5
                root=1;
            end
            obj.treeRoot=root;
            
            obj.treeStructure=tree;%Undirect2Direct(tree+tree');
            obj.treeIndices=treeindices;
            obj.directTree();
            
            %             assert(length(obj.treeRoot)==1);
            
        end
        function directTree(obj)
            %make sure the tree is directed
            tree=obj.treeStructure;
            directedTree=sparse(length(tree),length(tree));
            %Perform BFS on tree.
            %stack that holds nodes to visit
            roots=[obj.treeRoot];
            %perform bfs
            while(~isempty(roots)) %nodes in stack
                %pop node from stack
                root=roots(1);
                roots=roots(2:end);
                %find all nodes with edges to it
                sons=find(tree(root,:)|tree(:,root)');
                %make sure none of the children are in roots - that would
                %mean a cycle in the original undirected tree also
                assert(~any(ismember(sons,roots)));
                %insert all the children as children of the current node
                directedTree(root,sons)=1;
                %delete the adjacencies between children and current node
                %(so as to not make double edges when visiting children)
                tree(root,sons)=0;
                tree(sons,root)=0;
                %add children to nodes to visit
                roots=[roots sons];
            end
            obj.treeStructure=directedTree;
        end
        function cutTree(obj)
            if obj.alreadyCut
                error('can only cut once!');
            end

            obj.alreadyCut=1;
            obj.cutTreeRecurse(obj.treeRoot);

        end
        function cutTreeRecurse(obj,root)
            
            sons=find(obj.treeStructure(root,:));
            if isempty(sons)
                return;
            end
            
            starPaths={};
            sourceInd=obj.treeIndices(root);
            for i=1:length(sons)
                son=sons(i);
                targetInd=obj.treeIndices(son);
                %computing weighted adjacencies
                E=[obj.T(:,[1 2]);obj.T(:,[1 3]);obj.T(:,[2 3])];
                E=sort(E,2);
                E=unique(E,'rows');
                d=obj.V(E(:,1),:)-obj.V(E(:,2),:);
                d=sqrt(sum(d.^2,2));
                A=sparse([E(:,1);E(:,2)],[E(:,2);E(:,1)],[d;d],length(obj.V),length(obj.V));
                %remove boundary indices
                tri=triangulation(obj.T,obj.V);
                binds=tri.freeBoundary();
                if ~isempty(binds)
                    binds=setdiff(binds(:,1),[sourceInd targetInd]);
                    A(binds,:)=0;
                    A(:,binds)=0;
                end
                [~,newPath]=graphshortestpath(A,sourceInd, targetInd);
                newPath=newPath(1:end);
                starPaths{end+1}=obj.split_mesh_by_path(newPath);
                obj.finishedPaths=obj.finishedPaths+1;
            end
            obj.splitCenterNode(obj.treeIndices(root),starPaths);
            
            for i=1:length(sons)
                son=sons(i);
                obj.cutTreeRecurse(son);
            end
            
        end
        function  splitCenterNode(obj,center ,starPathPairs)
            %after splitting a "star", that is all sons of a current root
            %node, we need to duplicate the root several times, as it is
            %not duplicated during the actual cutting.
            %center - index of the root of the "star"
            %starPathPairs - the pathPairs of the star
            
            
            %find all tris touching the center vertex
            inds=find(any(ismember(obj.T,center),2));
            %now gonna split the one-rign to groups of adjacent tris
            groups={};
            %inds is the stack of tris to assign to a group
            while(true)
                %get the first tri from the stack
                theGroup=inds(1);
                %now expand the group from the seed
                while(true)
                    %get all vertices in current tri group
                    vs=unique(obj.T(theGroup,:));
                    %remove the center
                    vs=setdiff(vs,center);
                    %find all tris in the one ring that have a vertex in the group (not
                    %the center).
                    newMembers=find(any(ismember(obj.T(inds,:),vs),2));
                    %if exhausted all tris, stop
                    if isempty(newMembers)
                        break
                    end
                    %if found new members add them to group
                    theGroup=[theGroup;inds(newMembers)];
                    %and remove them from the stack
                    inds(newMembers)=[];
                end
                %add the new group
                groups{end+1}=unique(theGroup);
                %if handled all tris in one-ring, finish.
                if isempty(inds)
                    break;
                end
                
                
            end
            
            %now insert copies of the center tri and update the adjacencies
            group_centers={};
            for i=1:length(groups)
                %current group
                g=groups{i};
                %tris in current group
                t=obj.T(g,:);
                %if it's the first group no need to assign a new ind, we will just use
                %the existing one (so it is assigned to one group)
                if i>1
                    %insert copy of center
                    obj.V=[obj.V;obj.V(center,:)];
                    centerInd=length(obj.V);
                    obj.cutIndsToUncutInds(centerInd)=center;
                    obj.uncutIndsToCutInds{center}=[obj.uncutIndsToCutInds{center} centerInd];
                else
                    centerInd=center;
                end
                %update all instances of original vertex with the new one
                t(t==center)=centerInd;
                group_centers{i}=centerInd;
                obj.T(g,:)=t;
                %correct the paths
                
            end
            
            for j=1:length(starPathPairs)
                %get a pair of (coreposidning) paths
                pair=starPathPairs{j};
                %for each of the pair
                
                centers=nan(1,2);
                for k=1:2
                    for i=1:length(groups)
                        %current group
                        g=groups{i};
                        %if this half of the pair is in g it should get the new ind.
                        %since the star paths do not contain the centerVertex if they
                        %share a member with the group it must be some other vertex
                        %than the center one.
                        if any(ismember(pair(:,k),obj.T(g,:)))
                            if ~isnan(centers(k))
                                error('something is wrong');
                            end
                            centers(k)=group_centers{i};
                        end
                    end
                    
                end
                assert(~any(isnan(centers)));
                pair=[centers;pair];
                starPathPairs{j}=pair;
            end
            
            
            for j=1:length(obj.pathPairs)
                %get a pair of (coreposidning) paths
                pair=obj.pathPairs{j};
                %for each of the pair
                %%% TODO make eaceh pair correspond to group and then assign the
                %%% new center according to that.
                centers=nan(1,2);
                for k=1:2
                    for i=1:length(groups)
                        %current group
                        g=groups{i};
                        %if this half of the pair is in g it should get the new ind
                        %for the old paths we check all inds except for the last one,
                        %as the last one cannot be a member of the groups unless its
                        %the center of the star, in which case it being a member is not
                        %indicative to which group this path belongs
                        if any(ismember(pair(1:end-1,k),obj.T(g,:)))
                            assert(isnan(centers(k)));
                            centers(k)=group_centers{i};
                        end
                    end
                    
                end
                assert(isnan(centers(1))==isnan(centers(2)));
                %if nan means this path is not part of the star - nothing to do
                if ~isnan(centers(1))
                    
                    pair(end,:)=centers;
                    obj.pathPairs{j}=pair;
                end
            end
            obj.pathPairs=[obj.pathPairs starPathPairs];
        end
        function [ two_tris] = newTrisToInsert(obj, tri,shared_edge,ind_to_insert)
            %find the index that is not part of the edge we are to split
            otherind=setdiff(tri,shared_edge);
            %find the place of the ind
            indplace=find(tri==otherind);
            %set the tri s.t. the other ind is first and the edge to split is in [2 3]
            tri=tri([(indplace):3 1:(indplace-1)]);
            %create the two tris: [new e1 split] and [new e3 split
            two_tris=[tri([1 2]) ind_to_insert;ind_to_insert tri([3 1]) ];
            for i=1:2
                assert(length(unique(two_tris(i,:)))==3);
            end
        end
        
        
        function [path_corr ] = split_mesh_by_path( obj,p )
            %split the mesh by a given list of indices that describe a list
            %of adjacent edges to cut.
            
            
            %%%TODO - need to check if crossing existing edge on other paths,
            %%%if so need to refuse split
            
            %will hold which tris are to left\right of cut
            left=[];
            right=[];
            %go over the entire path
            for j=1:length(p)-1
                %the next edge to check
                e=[p(j:j+1)];
                %find the two tris that are adjacent to it
                tris_to_split=find(sum(ismember(obj.T,e),2)==2);
                
                assert(length(tris_to_split)==2);
                    
                %take the 1st tri to split of the pair
                tri=obj.T(tris_to_split(1),:);
                %check its orientation wrt the edge
                ind1= find(tri==e(1));
                ind2=find(tri==e(2));
                inds=[ind1 ind2];
                %positive orientation
                if all(inds==[1 2]) || all(inds==[2 3]) || all(inds==[3 1])
                    left=[left;tris_to_split(1)];
                    right=[right;tris_to_split(2)];
                else% negative orientation
                    left=[left;tris_to_split(2)];
                    right=[right;tris_to_split(1)];
                end
            end
            %now find tris that touch ANY vertex on the path that's not an end
            %point
            inds=find(any(ismember(obj.T,p(2:end-1)),2));
            %remove from these the tris we already found to be adjacent to edges
            inds=setdiff(inds,left);
            inds=setdiff(inds,right);
            for iter=1:1000
                %find all tris adjacent to a tri on the right side
                for j=1:length(right)
                    r=find(sum(ismember(obj.T(inds,:),obj.T(right(j),:)),2)>=2);
                    right=[right;inds(r)];
                end
                %find all tris adjacent to a tri on the left side
                for j=1:length(left)
                    l=find(sum(ismember(obj.T(inds,:),obj.T(left(j),:)),2)>=2);
                    left=[left;inds(l)];
                end
                %make sure left and right are adjoint
                right=setdiff(right,left);
                %remove the found tris from the inds
                inds=setdiff(inds,right);
                inds=setdiff(inds,left);
                %if finished all touching tris we can finish
                if isempty(inds)
                    break
                end
            end
            
            
            %will hold the correspondences beteween the two sides of the seam
            cur_path_corr=[];
            %go over all points not an end point
            for j=2:length(p)-1
                %duplicate vertex
                new_pathV=obj.V(p(j),:);
                obj.V=[obj.V;new_pathV];
                
                
                newInd=length(obj.V);
                %we change the indices of all tris on the left side of the cut
                tleft=obj.T(left,:); %take the tris
                tleft(tleft==p(j))=newInd; %replace the ind
                obj.T(left,:)=tleft; %insert tris back
                cur_path_corr=[cur_path_corr;p(j),newInd]; %insert new pair into correspondance
                obj.uncutIndsToCutInds{p(j)}=[obj.uncutIndsToCutInds{p(j)} newInd];
                obj.cutIndsToUncutInds(newInd)=p(j);
                
            end
            %add the last vertex on path. We do not split it, but we need it to keep
            %track of which vertices are on which edge
            cur_path_corr=[cur_path_corr;p(end) p(end)];
            
            path_corr=cur_path_corr;
            % patch('Faces',T(right,:),'Vertices',V,'FaceColor','blue');
            % patch('Faces',T(left,:),'Vertices',V,'FaceColor','red');
        end
        function  visualize(obj)
            hold on
            
            patch('Faces',obj.T,'Vertices',obj.V,'FaceColor','White');
            % set(gcf,'units','normalized','outerposition',[0 0 1 1])
            c_lim=2;
            for i=1:length(obj.pathPairs)
                
                p=obj.pathPairs{i};
                %scatter(obj.V(p(1,1),1),obj.V(p(1,1),2),80,'filled');
                if size(obj.V,2)==2
                    scatter(obj.V(p(end,1),1),obj.V(p(end,1),2),120,'filled');
                    scatter(obj.V(p(end,2),1),obj.V(p(end,2),2),120,'filled');
                    scatter(obj.V(p(1,1),1),obj.V(p(1,1),2),120,'filled');
                    scatter(obj.V(p(1,2),1),obj.V(p(1,2),2),120,'filled');
                    if nargin>4
                        c=lineCol;
                    else
                        c=hsv2rgb([i/length(obj.pathPairs),1,1]);
                    end
                    line(obj.V(p(:,1),1),obj.V(p(:,1),2),'linewidth',4,'Color',c);
                    line(obj.V(p(:,2),1),obj.V(p(:,2),2),'linewidth',4,'Color',c);
                    
                    scatter(obj.V(p(:,1),1),obj.V(p(:,1),2),20,'filled','black');
                    scatter(obj.V(p(:,2),1),obj.V(p(:,2),2),20,'filled','black');
                    
                    %scatter(obj.V(p(:,2),1),obj.V(p(:,2),2),'linewidth',3,'Color',hsv2rgb([i/length(obj.pathPairs),1,1]));
                    
                else
                    scatter3(obj.V(p(end,1),1),obj.V(p(end,1),2),obj.V(p(end,1),3),120,'filled');
                    scatter3(obj.V(p(end,2),1),obj.V(p(end,2),2),obj.V(p(end,2),3),120,'filled');
                    
                    line(obj.V(p(:,1),1),obj.V(p(:,1),2),obj.V(p(:,1),3),'linewidth',3,'Color',hsv2rgb([i/length(obj.pathPairs),1,1]));
                    line(obj.V(p(:,2),1),obj.V(p(:,2),2),obj.V(p(:,1),3),'linewidth',3,'Color',hsv2rgb([i/length(obj.pathPairs),1,1]));
                end
            end
            
            caxis([1 c_lim])
            axis equal
            %saveas(h,sprintf('%d.png',i),'png');
        end
    end
end

