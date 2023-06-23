function [] = savedata(surface_3D, source_3D, target, prior_a, prior_D, mouse,...
                       save_path, sign, rotate_sign, angle, set_num)
                   
if rotate_sign == 1
    surface_3D = imrotate(surface_3D, angle(set_num), 'crop');
    source_3D = imrotate(source_3D, angle(set_num), 'crop');
    target = imrotate(target, angle(set_num), 'crop');    
    prior_a = imrotate(prior_a, angle(set_num), 'crop');
    prior_D = imrotate(prior_D, angle(set_num), 'crop');
    mouse = imrotate(mouse, angle(set_num), 'crop'); 
end    

voxel_1 = surface_3D;
voxel_2 = source_3D;    
voxel_3 = target; 
voxel_4 = prior_a;    
voxel_5 = prior_D; 
voxel_6 = mouse * 0.1;    

voxel_1(voxel_1<0) = 0;
voxel_2(voxel_2<0) = 0;
voxel_3(voxel_3<0) = 0;
voxel_4(voxel_4<0) = 0;
voxel_5(voxel_5<0) = 0;
voxel_6(voxel_6<0) = 0;

save([save_path num2str(sign) '.mat'], 'voxel_1', 'voxel_2', 'voxel_3', 'voxel_4', 'voxel_5', 'voxel_6');

end

