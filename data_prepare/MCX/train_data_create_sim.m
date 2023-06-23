save_path = './data1000/';
read_path = './';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

k = 1;
bar = waitbar(0,'computing...');
% path, body: 380 208 350; brain: 180 208 300; 1: sampling, 2 2 4 pooling
[mouse,~] = rawread([read_path 'brain_180_208_300.raw'], 180, 208, 300, 3, 4, 5, 1);
%mouse = voxel_3;

[x,y,z] = ind2sub(size(mouse),find(mouse~=0));
[prop, prior_D, prior_a] = optical_parameters(mouse);
sign = 1; % 1 single source; 2 dual source
single = 0;
dual = 0;

set_num = 1000;  % number of training data
angle_list = randperm(360);
angle = angle_list(1:100);
rotate_sign = 0;    

for i = 1:set_num
    
    waitbar(i/set_num,bar,['computing...',num2str(100*i/set_num),'%']);
    newcfg.vol = mouse;    
    newcfg.prop = prop;
    newcfg.srctype='pattern3d'; 
    newcfg.unitinmm = 0.1;
    newcfg.tstart = 0;
    newcfg.tend = 5e-1;
    newcfg.tstep = 5e-1;
     
    %switch sign
    %   case 1
    %        sign = 2;
    %   case 2
    %        sign = 1;
    %end

    CWfluencem = NaN;
    CWfluencem_1 = NaN;
    CWfluencem_2 = NaN;

    if sign == 1
        while isnan(CWfluencem)

            Rsrc = 4; % radius of source, 4 voxel
            target_x = randi([min(x)+Rsrc,max(x)-Rsrc], 1); % mass of center
            target_y = randi([min(y)+Rsrc,max(y)-Rsrc], 1);
            target_z = randi([min(z)+Rsrc,max(z)-Rsrc], 1);  
            shape = randi([1,3], 1); % source shape: 1: ball 2: cylindar 3: cubic
%             target_x = 45;
%             target_y = 54;
%             target_z = 35;  
%             shape = 1;                                               

            [CWfluencem,target,source] = single_source_create(target_x, target_y, target_z, Rsrc, shape, newcfg);%�޸�
        end
        [source_3D,target] = normal(CWfluencem,target);
        single =single + 1;
    else
        while isnan(CWfluencem_1) | isnan(CWfluencem_2) | norm([target_x_1, target_y_1, target_z_1] - [target_x_2, target_y_2, target_z_2]) <= (Rsrc + Rsrc) + 10
            
            Rsrc = 4; % radius of source, 4 voxel
            target_x_1 = randi([min(x)+Rsrc,max(x)-Rsrc], 1);
            target_y_1 = randi([min(y)+Rsrc,max(y)-Rsrc], 1);
            target_x_2 = randi([min(x)+Rsrc,max(x)-Rsrc], 1);
            target_y_2 = randi([min(y)+Rsrc,max(y)-Rsrc], 1);
            target_z_1 = randi([min(z)+Rsrc,max(z)-Rsrc], 1); 
            target_z_2 = randi([min(z)+Rsrc,max(z)-Rsrc], 1);            
            shape = randi([1,3], 1);          
%           [25,65], [30,60], [35,55]
%             target_x_1 = 25;
%             target_y_1 = 54;
%             target_x_2 = 65;
%             target_y_2 = 54;
%             target_z = 35;
%             shape = 1;

            if norm([target_x_1, target_y_1, target_z_1] - [target_x_2, target_y_2, target_z_2]) <= (Rsrc + Rsrc)
                continue;           
            else
                [CWfluencem_1,target_1,source_1] = single_source_create(target_x_1, target_y_1, target_z_1, Rsrc, shape, newcfg);
                [CWfluencem_2,target_2,source_2] = single_source_create(target_x_2, target_y_2, target_z_2, Rsrc, shape, newcfg);
            end
        end
        [source_3D_1,target_1] = normal(CWfluencem_1,target_1);
        [source_3D_2,target_2] = normal(CWfluencem_2,target_2);
        source_3D = source_3D_1 + source_3D_2; % add two sources
        target = target_1 + target_2;
        source = [source_1,source_2];
        dual = dual + 1;
    end
    [surface_3D] = getsurface(newcfg, source_3D) .* source_3D;
    % surface_3D -> source_3D -> target: dice 0.6
    savedata(surface_3D, source_3D, target, prior_a, prior_D, mouse, save_path, i, rotate_sign, angle); % 6vol:
    save([save_path num2str(i) '_sign.mat'], 'sign')
    k = k + 1;  
end
disp(single);
disp(dual);
close(bar);
