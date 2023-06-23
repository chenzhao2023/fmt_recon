figure(2);

surface = x;
predict = pred;
source = y;
fym = x2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[X,Y] = meshgrid(1.0:1:64,...
                 1.0:1:64);
%
% for i = 1:0.1:64
%     [Z] = ones(size(X, 1),size(Y, 2)) .* i;
%     C = fym(:,:,round(i));
%     C(C==0) = nan;
%     mesh(X,Y,Z,C);
% %     surf(X,Y,-Z + 70,C);
%     shading interp;
%     hold on;
%     alpha(0.0001);
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

surface(surface~=0)=1;
for i = 1:64
    if sum(sum(predict(:,:,i))) > 64*60
        predict(:,:,i) = zeros(size(predict(:,:,i)));
    end
    surface(:,:,i) = imfill(surface(:,:,i),"holes");
%     predict(:,:,i) = medfilt2(predict(:,:,i),[3,3]);
end

patch(isosurface(surface,0),'FaceColor',[255/255,255/255,255/255],'FaceAlpha',0.1,...
	'EdgeColor','none');
patch(isosurface(predict,0),'FaceColor',[255/255,0/255,0/255],'FaceAlpha',0.5,...
	'EdgeColor','none');
patch(isosurface(source,0),'FaceColor',[0/255,0/255,255/255],'FaceAlpha',0.5,...
	'EdgeColor','none');
set(gcf,'unit','normalized','position', [0.2,0.2,0.28,0.48]);
set(gca,'ZDir','reverse')
colormap('jet');
view(225,45)
set(gca,'FontName','Times New Roman','FontSize',30,'FontWeight','bold')
set(gca,'linewidth',1.75)
set(gca,'box','off')
grid off;
box on;
colormap('jet');
set(gcf,'color','black');
colordef black;


OR = 0;
ADD = 0;
for x = 1:64
    for y = 1:64
        for z = 1:64
            if source(x,y,z) ~=0 && predict(x,y,z) ~=0
                ADD = ADD + 1;
            end
            if source(x,y,z) ~=0
                OR = OR + 1;
            end
            if predict(x,y,z) ~=0
                OR = OR + 1;
            end
        end
    end
end
DICE = 2*ADD/OR;

[x_1,y_1,z_1] = ind2sub(size(source),find(source~=0));
[x_2,y_2,z_2] = ind2sub(size(predict),find(predict~=0));
LE = norm(mean([x_1,y_1,z_1])-mean([x_2,y_2,z_2]))*0.1;