function [V] = smoothCones(fInfo,V)

cones = cell2mat(cellfun(@(x) x.flattener.inds, fInfo,'uniformoutput',false));
uCones = unique(cones);
vring = compute_vertex_ring(fInfo{1}.flattener.M_orig.F');
for ii=1:length(uCones)
    V(uCones(ii),:) = mean(V(vring{uCones(ii)},:));
end

end

