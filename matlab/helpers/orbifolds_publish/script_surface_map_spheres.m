%% =====================================================================
%  ===                      Setting up stuff                         ===
%  =====================================================================

%Read the mesh
[V1,T1]=read_off('max_map.off');V1=V1';T1=T1';
[V2,T2]=read_off('sphere.off');V2=V2';T2=T2';
%Some color-related variables - no need to concern yourself with these :)
cone_colors=[1 0.8 0;0.7 0 1; 0 0.5 0.8;0 0 0.5];
cut_colors=[1 0 0;0 1 0;0 0 1];

%The cone positions and the choice of orbifold structure, defining the
%desired embedding

% === TYPE I orbifold ===
% cones1=[1 10000 20000];
% cones2=[20 120 180];
% orbifold_type=OrbifoldType.Square;

%try uncommneting each of the next lines for the other orbifold structures

% === TYPE II orbifold ===
% cones=[50 100 130];
% orbifold_type=OrbifoldType.Diamond;

% === TYPE III orbifold ===
% cones=[50 100 130];
% orbifold_type=OrbifoldType.Triangle;

% === TYPE IV orbifold ===
% cones=[50 100 130 140];
orbifold_type=OrbifoldType.Square;

cones1=[  25372
    12611
    5868
    ];
cones2=[  10
    20
    30
    ];



%% =======================================================================
%  =======       The actual algorithm! cutting and flattening      =======
%  =======================================================================



% [V_flat1,cutMesh1]=flatten_sphere(V1,T1,cones1,orbifold_type);
% [V_flat2,cutMesh2]=flatten_sphere(V2,T2,cones2,orbifold_type);
% BC=compute_map_from_sphere_embeddings( V_flat1,V_flat2,cutMesh1,cutMesh2 );
BC=map_spheres(V1,T1,V2,T2,cones1,cones2,orbifold_type);
V_mapped=BC*V2;
%%
tr=triangulation(T1,V1);
VN=tr.faceNormal();
VN=(VN+1)/2;
VN(all(VN==0,2),:)=1;
VN=normr(VN);
% VN=V1;
% for i=1:3
%     VN(:,i)=VN(:,i)-min(VN(:,i));
%     VN(:,i)=VN(:,i)/max(VN(:,i));
% end
% VN(isnan(VN))=0;
% VN=VN*5;
% VN=mod(VN,1);
% VN(:,3)=0;
% % VN=repmat(VN(:,2),1,3);
% % VN(:,[3 ])=0;

fig=figure(1);
    clf

set(gcf, 'Position', get(0,'Screensize'));

% subplot(1,2,2);
% while(true)
% error('maximize figure');
M=struct('cdata',[],'colormap',[]);
step=0.02;
count=1;
ax=[];
for c=[0:step:1-step 1:-step:0]
    c
    clf
    V=V_mapped*c+(1-c)*V1;
    patch('faces',T1,'vertices',V,'facecolor','flat','edgecolor','k','edgealpha',1,'FaceVertexCDATA',VN);
    campos([-39.5547   51.5112 -130.9298]);
    camup([0.1032    0.9349    0.3395]);
    axis([ -10.4249    8.5982   -6.1319    5.0566   -7.3091    6.6766])
    campos([-39.5547   51.5112 -130.9298]);
    camup([0.1032    0.9349    0.3395]);
    % campos([58.6269   45.4568 -126.0013]);
    % camup([ -0.1328    0.9493    0.2850]);
    axis off
    camlight
    hold on
    fig.PaperPositionMode = 'auto';
    for i=1:length(cones2)
        VV=V(cones1(i),:);
        scatter3(VV(:,1),VV(:,2),VV(:,3),60,cone_colors(i,:),'fill');
    end
    drawnow
    pause(0.001);
%     print(sprintf('%04d.png',count),'-dpng','-r200');
    % break
    count=count+1;
    
    %               M(end+1) = getframe;
end
M(1)=[];
figure(1);
% while(true)
% figure();
movie(M);
% end
% pause(1);
% end
% hold on
% for i=1:length(cones2)
%     VV=V2(cones2(i),:);
%     scatter3(VV(:,1),VV(:,2),VV(:,3),40,cone_colors(i,:),'fill');
% end
% axis equal
% subplot(1,2,1);
% patch('faces',T1,'vertices',V1,'facecolor','interp','edgecolor','none','FaceVertexCDATA',VN);
% axis equal