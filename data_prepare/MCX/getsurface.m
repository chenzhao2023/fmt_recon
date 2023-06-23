function [surface] = getsurface(cfg,CWfluencem)
surface = zeros(size(cfg.vol));
[~,~,z] = ind2sub(size(cfg.vol),find(cfg.vol~=0));

surface(:,:,min(z)) = cfg.vol(:,:,min(z));
surface(:,:,max(z)) = cfg.vol(:,:,max(z));

surface(surface~=3) = 0; % add this line or delete;

for z = 2:size(cfg.vol, 3)-1
    for x = 1:size(cfg.vol, 1)
        for y = 1:1:size(cfg.vol, 2)
            if CWfluencem(x,y,z) ~= 0 && surface(x,y,z) == 0
                surface(x,y,z) = 1;
                break;
            end
        end
    end
end

for z = 2:size(cfg.vol, 3)-1
    for y = 1:size(cfg.vol, 2)
        for x = 1:1:size(cfg.vol, 1)
            if CWfluencem(x,y,z) ~= 0 && surface(x,y,z) == 0
                surface(x,y,z) = 1;
                break;
            end
        end
    end
end

for z = 2:size(cfg.vol, 3)-1
    for x = size(cfg.vol, 1):-1:1
        for y = size(cfg.vol, 2):-1:1
            if CWfluencem(x,y,z) ~= 0 && surface(x,y,z) == 0
                surface(x,y,z) = 1;
                break;
            end
        end
    end
end

for z = 2:size(cfg.vol, 3)-1
    for y = size(cfg.vol, 2):-1:1
        for x = size(cfg.vol, 1):-1:1
            if CWfluencem(x,y,z) ~= 0 && surface(x,y,z) == 0
                surface(x,y,z) = 1;
                break;
            end
        end
    end
end

surface(surface~=0) = 1;

end

