function [prop, prior_D, prior_a] = optical_parameters(mouse)
prop=[0      0        1    1;
     0.07859 6.710400 0.85 1.37;%heart
     0.01504 18.49734 0.92 1.37;%stomach
     0.47078 6.999990 0.90 1.37;%liver
     0.08811 16.84647 0.86 1.37;%kidney
     0.26296 36.81805 0.94 1.37;%lung
     0.11636 4.673520 0.90 1.37;];%muscle    650nm
% ---- line 10 to 13 for brain --------------
prop=[0      0     1   1;
      0.07515 4.115140 0.90 1.37;%muscle
      0.05251 24.41529 0.9 1.37;%skull
      0.0183 10.90 0.9 1.37;]; %brain
 
% prop=[0       0      1   1;
%       0.07515 4.115140 0.90 1.37;]; %cube/cylinder 
% ---- line 18 to 25: body --------------
% prop=[0      0        1    1;
%      0.07515 4.115140 0.90 1.37;%muscle   
%      0.05087 6.291100 0.85 1.37;%heart
%      0.00992 17.70523 0.92 1.37;%stomach
%      0.30413 6.676000 0.90 1.37;%liver
%      0.05708 15.73691 0.86 1.37;%kidney
%      0.16955 35.94803 0.94 1.37;%lung
%      ]; %680nm

u_a = prop(2:end,1);
D =  (3 * (prop(2:end,1) + (1 - prop(2:end,3)) .* prop(2:end,2))).^-1;

prior_D = round(mouse);
prior_a = round(mouse);

for i = 1:length(D)
    prior_D(prior_D == i) = D(i);
    prior_a(prior_a == i) = u_a(i);
end

