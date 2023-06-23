function [source_3D,target] = normal(CWfluencem,target)
    source_3D = CWfluencem ./ max(max(max(CWfluencem)));
    target(target~=0) = 1;
end

