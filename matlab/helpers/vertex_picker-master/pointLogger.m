classdef pointLogger < handle
    %POINTLOGGER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        points=zeros(0,3);
        inds=[];
        curInd;
        curPoint;
        handles=[];
        name;
    end
    
    methods
        function obj=pointLogger(name)
            if nargin>0
                obj.name=name;
            end
        end
        function setCurPoint(obj,point,ind)
            obj.curPoint=point;
            obj.curInd=ind;
        end
        function added=addCurPoint(obj)
            added=false;
            if isempty(obj.curPoint) || ismember(obj.curInd,obj.inds)
                return;
            end
            added=true;
            obj.points(end+1,:)=obj.curPoint;
            obj.inds(end+1)=obj.curInd;
            obj.curPoint=[];
            obj.curInd=[];
        end
        function removeLastPoint(obj)
            if ~isempty(obj.points)
                obj.points(end,:)=[];
                obj.inds(end)=[];
            end
            obj.curPoint=[];
            obj.curInd=[];
        end
        function draw(obj)
            for i=1:length(obj.handles)
                try
                    
                    delete(obj.handles(i));
                catch
                end
            end
            
            obj.handles=[];
            if ~isempty(obj.points)
                obj.handles(end+1) = plot3(obj.points(:,1),obj.points(:,2),obj.points(:,3), 'blackO', 'MarkerSize', 10);
                obj.handles(end+1) = plot3(obj.points(:,1),obj.points(:,2),obj.points(:,3), 'blue.', 'MarkerSize', 30);
                for i=1:size(obj.points,1)
                    obj.handles(end+1) = text(obj.points(i,1),obj.points(i,2),obj.points(i,3),num2str(i),'fontsize',20);
                end
            end
            if ~isempty(obj.curPoint)
                obj.handles(end+1) = plot3(obj.curPoint(:,1),obj.curPoint(:,2),obj.curPoint(:,3), 'blackO', 'MarkerSize', 10);
                obj.handles(end+1) = plot3(obj.curPoint(:,1),obj.curPoint(:,2),obj.curPoint(:,3), 'black.', 'MarkerSize', 30);
                obj.handles(end+1) = plot3(obj.curPoint(:,1),obj.curPoint(:,2),obj.curPoint(:,3), 'blackX', 'MarkerSize', 20);
            end
        end
        function save(obj)
            assignin('base', 'selected_points', obj.inds);
            disp('the indices of the selected points were exported to the workspace variable ''selected_points''');
            if ~isempty(obj.name)
                name=obj.name;
                name=strsplit(name,'.');
                name=name{1};
                curname=[name '.mat'];
                if exist(curname)==2
                    for i=1:1000
                        curname=[name '_' num2str(i) '.mat'];
                        if exist(curname)~=2
                            break;
                        end
                    end
                end
                %                 SAVE_NAME=[name '_' datestr(now,'yymmddHHMM') '.mat'  ];
                inds=obj.inds;
                mesh_name=obj.name;
                save(curname,'inds','mesh_name');
                disp(['and saved to the file named ''' curname '''']);
            end
        end
    end
    
end

