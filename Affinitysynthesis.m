function [C] = Affinitysynthesis(U,V)

U_tmp = zeros(size(U{1}));
v = size(U,2);
for i = 1:v
    U_tmp = U_tmp+U{i};
end
U_tmp = U_tmp/v;
C_tmp = U_tmp*V;
C = U_tmp*V;
