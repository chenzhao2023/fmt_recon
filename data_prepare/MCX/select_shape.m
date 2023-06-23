function [sphsrc,target] = select_shape(target_x, target_y, target_z, Rsrc, shape, newcfg)
    [X,Y,Z] = size(newcfg.vol);
    target = zeros(X,Y,Z);
    disp(shape);
    if shape == 1
        %球体
        [xi,yi,zi]=ndgrid(-Rsrc:Rsrc,-Rsrc:Rsrc,-Rsrc:Rsrc);
        sphsrc=((xi.*xi + yi.*yi + zi.*zi)<=Rsrc*Rsrc);
        for x = 1:X
            for y = 1:Y
                for z = 1:Z
                    if norm([x,y,z]-[target_x,target_y,target_z]) <= Rsrc
                       target(x,y,z) = 1;
                    end
                end
            end
        end
    elseif shape == 2
        %圆柱
        [xi,yi,zi]=ndgrid(-5:5,-5:5,-5:5);
        sphsrc=((xi.*xi + yi.*yi)<=Rsrc*Rsrc);
        for x = 1:X
            for y = 1:Y
                for z = 1:Z
                    if norm([x,y]-[target_x,target_y]) <= Rsrc && abs(target_z-z) <= 5
                       target(x,y,z) = 1;
                    end
                end
            end
        end      
    elseif shape == 3
        %正方体
        [xi,yi,zi]=ndgrid(-Rsrc:Rsrc,-Rsrc:Rsrc,-Rsrc:Rsrc);
        sphsrc=(xi.*xi<=Rsrc*Rsrc);  
        for x = 1:X
            for y = 1:Y
                for z = 1:Z
                    if abs(target_x-x) <= Rsrc && abs(target_y-y) <= Rsrc && abs(target_z-z) <= Rsrc
                       target(x,y,z) = 1;
                    end
                end
            end
        end        
    elseif shape == 4
        %自定义体
        target = newcfg.source;
        sphsrc=logical(target);       
    end
end

