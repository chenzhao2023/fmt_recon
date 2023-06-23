function [CWfluencem,target,source] = single_source_create(target_x, target_y, target_z, Rsrc, shape, newcfg)
    if target_x-Rsrc > 0 && target_y-Rsrc > 0 && target_z-Rsrc > 0 || Rsrc == 0
        if (newcfg.vol(target_x-Rsrc,target_y-Rsrc,target_z-Rsrc)~=0)&&(newcfg.vol(target_x+Rsrc,target_y+Rsrc,target_z+Rsrc)~=0)
            newcfg.srcpos = [target_x-Rsrc,target_y-Rsrc,target_z-Rsrc];
            newcfg.srcdir =[0,0,1,nan]; 
            [sphsrc,target] = select_shape(target_x, target_y, target_z, Rsrc, shape, newcfg);
            newcfg.srcpattern = sphsrc;
            newcfg.srcparam1=size(newcfg.srcpattern);
            newcfg.issrcfrom0=1;
            newcfg.isspecula=0;
            newcfg.nphoton=1e6;
            fluem=mcxlab(newcfg);
            CWfluencem=fluem.data;
            source = [target_x,target_y,target_z,Rsrc];
        else
            source = NaN;
            CWfluencem = NaN;
            target = NaN;
            shape = NaN;
        end
    else
        source = NaN;
        CWfluencem = NaN;
        target = NaN;
        shape = NaN;        
    end
end

