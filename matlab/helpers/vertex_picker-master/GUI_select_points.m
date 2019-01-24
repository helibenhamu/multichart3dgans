function logger = GUI_select_points(V,T,name)
clear logger;
figure(5);
close(5);
figure(5);
clf
if nargin<3
    name=[];
end
disp('click on a point on the mesh to select it');
disp('click ''n'' to progress to adding a *n*ew point while saving the prev');
disp('click ''d'' to *d*elete the last inserted point');
disp('click ''s'' to *s*ave and print the selected points to workspace');



% show the point cloud

hold off
% plot3(V(1,:), V(2,:), V(3,:), 'c.');
h=patch('vertices',V,'faces',T,'FaceColor',[0.6 0.6 0.6],'EdgeColor','k');
%light('Position',[1 0 0],'Style','local','color','cyan')
%light('Position',[0 1  0],'Style','local','color','magenta')
%light('Position',[0 0 1],'Style','local','color','yellow')
h.FaceLighting = 'flat';
h.AmbientStrength = 0.0;
h.DiffuseStrength = 0.8;
h.SpecularStrength = 0.0;
h.SpecularExponent = 25;
h.BackFaceLighting = 'lit';
cameratoolbar('Show');
hold on;
axis equal
logger=pointLogger(name);
tri=triangulation(T,V);
b=tri.freeBoundary();
if ~isempty(b)
    valid=b(:,1);
    
else
    valid=1:length(V);
end
hf=gcf;
set(h,'ButtonDownFcn',@callback_patch_click,...
    'PickableParts','visible');
set(hf,'KeyPressFcn',@key_handler);
    function key_handler(h_obj,evt)
        if strcmp(evt.Key,'n')
            
            added=logger.addCurPoint();
            if added
                disp('new point!');
            else
                disp('no point added!');
            end
        elseif strcmp(evt.Key,'d')
            logger.removeLastPoint();
        elseif strcmp(evt.Key,'s')
            logger.save();
        else
            return;
        end
        logger.draw();
    end
    function callback_patch_click(src, eventData)
        p=eventData.IntersectionPoint;
        ind=knnsearch(V(valid,:),p);
        ind=valid(ind);
        logger.setCurPoint(V(ind,:),ind);
        logger.draw();
        fprintf('you clicked on point number %d\n', ind);
    end
end

