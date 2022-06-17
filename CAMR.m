function [U,V,max_iter] = CAMR(X,opts)
% CAMR - Clean affinity matrix learning with rank equality constraint for multi-view subspace clustering
% PR-D-21-01858
%% Initialization variables
v = opts.v;
n = opts.n;

lambda_1 = opts.lambda_1;
lambda_2 = opts.lambda_2;
dim_V = opts.dim_V;
rho = opts.rho;

mu = 1e-4;
max_mu = 1e6;

E_x = cell(1,v);
E_z = cell(1,v);
U = cell(1,v);
Z = cell(1,v);
C = cell(1,v);
V = zeros(dim_V,n);
L = zeros(dim_V,n);
Y_1 = cell(1,v);
Y_2 = cell(1,v);
Y_3 = cell(1,v);
Y_4 = zeros(dim_V,n);

dim_all_view = zeros(v,1);
for i = 1:v
    dim_all_view(i) = size(X{i},1);
    E_x{i} = zeros(dim_all_view(i),n);
    E_z{i} = zeros(dim_all_view(i),n);
    U{i} = zeros(n,dim_V);
    Z{i} = zeros(n,n);
    C{i} = zeros(n,n);
    Y_1{i} = zeros(dim_all_view(i),n);
    Y_2{i} = zeros(n,n);
    Y_3{i} = zeros(n,n);
    Z{i} = rand(n,n);
end

%% Optimization
max_iter = 100;
iter_curr = 0;
conv_flag = 0;
conv_threshold = 1e-6;

while conv_flag==0 && iter_curr<max_iter
    
    % updating V:
    V_a = mu*eye(dim_V);
    V_b = mu*L-Y_4;
    for i = 1:v
        V_a = V_a + mu*U{i}'*U{i};
        V_b = V_b + U{i}'*Y_3{i} + mu*U{i}'*C{i};
    end
    V = V_a\V_b;
    
    % updating L:
    L_a = V + Y_4/mu;
    L = softth(L_a,lambda_2/mu);
    
    % updating variables that need to be updated for each views
    for i = 1:v
        % updating E_x
        E_a = X{i} - X{i}*Z{i} + Y_1{i}/mu;
        E_x{i} = solve_l1l2(E_a,1/mu);
        
        % updating E_z
        E_b = Z{i} - C{i} + Y_2{i}/mu;
        E_z{i} = solve_l1l2(E_b,lambda_1/mu);
        
        % updating U:
        U_a = C{i} + Y_3{i}/mu;
        U_b = V*U_a';
        [svd_U,~,svd_V] = svd(U_b,'econ');
        U{i} = svd_V*svd_U';
        
        % updating Z:
        Z_a = mu*X{i}'*X{i} + mu*eye(n);
        Z_b = X{i}'*Y_1{i} + mu*(X{i}'*X{i}-X{i}'*E_x{i}) -Y_2{i} + mu*(C{i}+E_z{i});
        Z{i} = Z_a\Z_b;
        
        % updating C:
        C_a = 2*mu*eye(n);
        C_b = Y_2{i}-Y_3{i}+mu*(Z{i}-E_z{i}+U{i}*V);
        C{i} = C_a\C_b;
        
        % updating Y_1 Y_2 and Y_3:
        Y_1{i} = Y_1{i} + mu*(X{i} - X{i}*Z{i} - E_x{i});
        Y_2{i} = Y_2{i} + mu*(Z{i} - C{i} - E_z{i});
        Y_3{i} = Y_3{i} + mu*(C{i} - U{i}*V);
    end
    
    % updating Y_4:
    Y_4 = Y_4 + mu*(V - L);
    
    % updating mu:
    mu = min(rho*mu,max_mu);
    
    % check the convergence conditions
    conv_count_tmp = 0;
    for i = 1:v
        if norm(X{i}-X{i}*Z{i}-E_x{i},inf) < conv_threshold
            conv_count_tmp = conv_count_tmp + 1;
        end
        if norm(Z{i} - C{i} - E_z{i},inf) < conv_threshold
            conv_count_tmp = conv_count_tmp + 1;
        end
        if norm(C{i} - U{i}*V,inf) < conv_threshold
            conv_count_tmp = conv_count_tmp + 1;
        end
        if norm(V-L,inf) < conv_threshold
            conv_count_tmp = conv_count_tmp + 1;
        end
    end
    
    if conv_count_tmp == 3*v
        conv_flag = 1;
    end
    
    iter_curr = iter_curr + 1;
end