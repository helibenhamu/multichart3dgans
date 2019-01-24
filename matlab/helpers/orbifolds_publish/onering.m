classdef onering < handle
    %ONERING Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        tri_inds;
        generator;
        boundaryInds;
        TR;
        orgTR;
    end
    
    methods
        function visualize(obj)
            patch('faces',obj.TR.ConnectivityList,'vertices',obj.TR.Points,'facecolor','green');
        end
        function obj=onering(tri_inds,generator,orgTR)
            obj.tri_inds=tri_inds;
            obj.generator=generator;
            obj.TR=triangulation(orgTR.ConnectivityList(obj.tri_inds,:),orgTR.Points);
            obj.orgTR=orgTR;
            if length(tri_inds)==1 %covers inside triangle and boundary case
                b=[obj.TR.ConnectivityList obj.TR.ConnectivityList(1)]';
            elseif length(generator)==2 %internal edge case
                assert(length(tri_inds)==2);
                inds=setdiff(obj.TR.ConnectivityList(:),generator);
                assert(length(inds)==2);
                b=[inds(1); generator(1); inds(2); generator(2); inds(1)];
            else %vertex case
                assert(length(generator)==1);
                b=obj.TR.freeBoundary();
                b=[b(:,1);b(end,2)];
            end
            
            obj.boundaryInds=b;
        end
        function b=isEdgeRing(obj)
            b=length(obj.generator)==2;
        end
        function b=isVertexRing(obj)
            b=length(obj.generator)==1;
        end
        function b=isTriRing(obj)
            b=length(obj.generator)==3;
        end
        function tri_ind=pointInTri(obj,p)
            %             ti=obj.TR.pointLocation(p);
            %             if isnan(ti)
            %                 tri_ind=[];
            %             else
            %                 tri_ind=obj.tri_inds(ti);
            %             end
            %%place holder - change to above when 2014a installed
            tri_ind=[];
            
            for i=1:length(obj.tri_inds)
                T=obj.orgTR.ConnectivityList(obj.tri_inds(i),:);
                Tx=obj.orgTR.Points(T,1);
                Ty=obj.orgTR.Points(T,2);
                if inpolygon(p(1), p(2), Tx, Ty)
                    tri_ind=obj.tri_inds(i);
                    break;
                end
            end
            
        end
        function [int_p,int_e]=intersectWithBoundaryOld(obj,edgex,edgey)
            ring_bdry = obj.TR.Points(obj.boundaryInds,:);
            [xi,yi,Ii] = polyxpoly(edgex,edgey,ring_bdry(:,1),ring_bdry(:,2));
            int_p=[];
            int_e=[];
            %didn't inresect any edge
            if isempty(xi)
                
                return;
            end
            Ii=Ii(:,2);
           
            %take closest point to father (first intersection point)
            diff = [xi yi] - repmat([edgex(1) edgey(1)],length(xi),1);
            d = sum(diff.^2,2);
            %if one of the points is the father itself, make sure not
            %to take it
            d(d<1e-16) = inf;
            [minval,ind] = min(d);
            
            %             end
            if minval==inf %too close to father - no intersection.
                return;
            end
            xi = xi(ind);
            yi = yi(ind);
            Ii = Ii(ind);
            
            if Ii(1)==length(obj.boundaryInds)%this will happen if the
                %last point on polyline is crossed; happens due to the way
                %polyxpoly handles intersections with vertices
                m=0;
                v_ind=I1;
                error
            else
                
                %take indices of intersection edge on traget
                int_e=[obj.boundaryInds(Ii) obj.boundaryInds(Ii+1)];
                
                %if very close to one of the two vertices of the edge, take
                %vertex
                
                diff = obj.orgTR.Points(int_e,:) - repmat([xi yi],2,1);
                d = sum(diff.^2,2);
                [m,v_ind]=min(d);
            end
            if m<1e-6 %intersection lands on vertex
                int_e=obj.boundaryInds(v_ind);
                
            end
            int_p=[xi yi];
        end
        function [p,edge]=intersectWithBoundary(obj,edgex,edgey)
            p1=[edgex edgey];
            tris=obj.TR.ConnectivityList;
            %             edges=[tris(:,[1 2]);tris(:,[2 3]);tris(:,[3 1])];
            %             edges=sort(edges,2);
            %             edges=unique(edges,'rows');
            edges=[obj.boundaryInds(1:end-1) obj.boundaryInds(2:end)];
            crossPoints=zeros(size(edges,1),2);
            for i=1:size(edges,1)
                
                p2=obj.orgTR.Points(edges(i,:),:);
                %                 den=det([p1(2,:)-p1(1,:);p2(1,:)-p2(2,:)]');
                %                 t=det([p1(1,:)-p2(1,:);p2(1,:)-p2(2,:)]')/den;
                %                 s=det([p1(2,:)-p1(1,:);p1(1,:)-p2(1,:)]')/den;
                
                st=-(p1(1,:)-p2(1,:))/[p1(2,:)-p1(1,:);p2(1,:)-p2(2,:)];
                t=st(1); s=st(2);
                eps=1e-6;
                if t>=0-eps && t<=1+eps && s>=0-eps && s<=1+eps
                    x=(1-t)*p1(1,1)+t*p1(2,1);
                    y=(1-t)*p1(1,2)+t*p1(2,2);
                else
                    x=[];
                end
                %                 for debug
                %                 [x1,y1]=polyxpoly(p1(:,1),p1(:,2),p2(:,1),p2(:,2));
                %                 if isempty(x1)
                %                     assert(isempty(x));
                %                 else
                %                     assert(abs(x-x1)<1e-4);
                %                     assert(abs(y-y1)<1e-4);
                %                 end
                %                 end debug
                
                if isempty(x)
                    crossPoints(i,:)=inf;
                else
                    crossPoints(i,:)=[x y];
                end
            end
            dists=bsxfun(@minus,crossPoints,[edgex(2) edgey(2)]);
            dists=sum(dists.^2,2);
            [mindist,ind]=min(dists);
            edge=edges(ind,:);
            p=crossPoints(ind,:);
            if isinf(p(1))||norm(p-[edgex(1) edgey(1)])<1e-10
                p=[];
                edge=[];
            else
                ps=obj.orgTR.Points(edge,:);
                d=repmat(p,2,1)-ps;
                d=sum(d.^2,2);
                [m,ind]=min(d);
                if sqrt(m)<1e-10
                    edge=edge(ind);
                end
            end
        end
    end
    
end

