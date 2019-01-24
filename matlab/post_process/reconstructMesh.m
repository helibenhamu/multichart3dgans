function [V_rec, scale, norm_scale,aligned_charts] = reconstructMesh(data, fInfo, triplets_table,align_charts, smooth_cones, p,scale_method,weighting_method,k)
% reconstructMesh
% default - convex comb with norm 1.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% handle inputs
if nargin<9
    k=1;
    if nargin<8
        weighting_method = 'convex_comb';
        if nargin<7
            scale_method = 'original';
            if nargin<6
                p=1;
                if nargin<5
                    smooth_cones = true;
                    if nargin<4
                        align_charts = true;
                    end
                end
            end
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% align and pad charts
if align_charts
    data = align_charts_ST(data, triplets_table, 3);
end
aligned_charts = [data , data(1:end,1,:) ; data(1,1:end,:),data(1,1,:)];

% extract scale
scale = getScale(scale_method, fInfo);
mod_scale = scale.^p;

V_rec = zeros(size(mod_scale,1),3);

switch weighting_method
    case 'convex_comb'        
        norm_scale = bsxfun(@rdivide,mod_scale,sum(mod_scale,2));

    case 'k_hot'        
        [~ , I] = sort(mod_scale,2,'descend');
        row_ind  = colStack(repmat(1:size(mod_scale,1),k,1));
        col_ind = colStack(I(:,1:k)');

        k_scale = zeros(size(mod_scale));
        k_scale(sub2ind(size(mod_scale),row_ind,col_ind)) = mod_scale(sub2ind(size(mod_scale),row_ind,col_ind));

        norm_scale = k_scale./repmat(sum(k_scale,2),1,size(triplets_table,1));
end

flattener = {};
for ii = 1:size(triplets_table,1)
    flattener{ii} = fInfo{1,ii}.flattener;
    V_rec = V_rec + [norm_scale(:,ii).*flattener{ii}.liftImage(aligned_charts(:,:,3*ii-2)),...
        norm_scale(:,ii).*flattener{ii}.liftImage(aligned_charts(:,:,3*ii-1)),...
        norm_scale(:,ii).*flattener{ii}.liftImage(aligned_charts(:,:,3*ii))];
end

if smooth_cones
    V_rec = smoothCones(fInfo, V_rec);
end

end

