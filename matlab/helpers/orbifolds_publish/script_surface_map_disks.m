% load mesh2
% Z1=zeros(length(V),1);sin(V(:,1)*2.6).*sin(V(:,2)*5.1)*0.5.* exp(-V(:,1).^2 - V(:,2).^2);
% V1=[V Z1];
N=320;
[X, Y, Z]=meshgrid(1:N,1:N,0);
X=X(:)-1;
Y=Y(:)-1;
Z=Z(:);
V=[X Y];
V1=[V Z];
T=delaunay(X,Y);
T1=T;
clear V;
clear T;

load mesh2
Z2=zeros(length(V),1);%sin(V(:,1)*1).*sin(V(:,2)*7.3)*0.5.* exp(-V(:,1).^2 - V(:,2).^2);
V2=[V Z2];
T2=T;
clear V;
clear T;
%set boundary conditions
tri1=triangulation(T1,V1);
b1=tri1.freeBoundary();
b1=b1(:,1);
tri2=triangulation(T2,V2);
b2=tri2.freeBoundary();
b2=b2(:,1);

%Some color-related variables - no need to concern yourself with these :)
cone_colors=[1 0.8 0;0.7 0 1; 0 0.5 0.8;0 0 0.5];
cut_colors=[1 0 0;0 1 0;0 0 1;0 1 1];

%The cone positions and the choice of orbifold structure, defining the
%desired embedding

% === Triangle disk orbifold ===
boundary_inds2=[1 100 180];
cones2=b2(boundary_inds2);

boundary_inds1=floor(linspace(1,length(b1),4));
boundary_inds1=boundary_inds1(1:3);
cones1=b1(boundary_inds1);

%try uncommneting each of the next lines for the other orbifold structures

% === Square disk orbifold ===
% boundary_inds=[1 70 80 100 ];
% cones=b(boundary_inds);





%% =======================================================================
%  =======       The actual algorithm! cutting and flattening      =======
%  =======================================================================



% [V_flat1,segs1]=flatten_disk(V1,T1,cones1);
% [V_flat2,segs2]=flatten_disk(V2,T2,cones2);
% 
% BC = compute_map_from_disk_embeddings( V_flat1,T1,V_flat2,T2,segs1,segs2 );
 BC_1to2 = map_disks(V1,T1,cones1,V2,T2,cones2);

V_mapped=BC_1to2*V2;



%% =======================================================================
%  =====                        Visualization                        =====
%  =======================================================================



figure(1);
clf;


% === Visualization of the original 3D mesh + cones & cuts on it ===
subplot(1,2,1);
%draw the mesh
tilesize=8;
maxx=-inf(length(T1),1);
maxy=-inf(length(T1),1);
for i=1:3
    maxx=max(maxx,V1(T1(:,i),1));
    maxy=max(maxy,V1(T1(:,i),2));
end
isblack=(mod(maxx,tilesize*2)<tilesize)~=(mod(maxy,tilesize*2)<tilesize);

colors=[floor(maxx/tilesize) floor(maxy/tilesize) ones(length(T1),1) ];
for i=1:3
    colors(:,i)=colors(:,i)-min(colors(:,i));
    colors(:,i)=colors(:,i)./max(colors(:,i));
end
colors(isnan(colors))=0;
% colors=(colors+0.5)/1.5;
colors(:,3)=1-colors(:,1);
colors=1-colors;
colors(isblack,:)=0;

patch('Faces',T1,'Vertices',V1,'FaceColor','flat','FaceVertexCData',colors,'edgecolor','none');
hold on;
for i=1:length(cones1)
    scatter3(V1(cones1(i),1),V1(cones1(i),2),V1(cones1(i),3),150,cone_colors(i,:),'fill');
end


axis equal
axis off
subplot(1,2,2);
%draw the mesh
patch('Faces',T1,'Vertices',V_mapped,'FaceColor','flat','FaceVertexCData',colors,'edgecolor','none');
hold on;
for i=1:length(cones1)
    scatter3(V_mapped(cones1(i),1),V_mapped(cones1(i),2),V_mapped(cones1(i),3),150,cone_colors(i,:),'fill');
end
axis equal
axis off