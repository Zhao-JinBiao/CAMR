clear;clc;
load NGs
fprintf('CAMR on NGs...\n');

opts.v = size(X,2);
opts.n = size(X{1},2);
opts.numClust = size(unique(gt{1}),1);

for i = 1:opts.v
    X{i} = X{i}./repmat(sqrt(sum(X{i}.^2,1)),size(X{i},1),1);
end

parm1= [0.01 0.1 1 10 100 1000 10000];
parm2= [0.01 0.1 1 10 100 1000 10000];

ldim = int_mod(opts.n , opts.numClust);

opts.rho = 1.9;

for e = 1:length(ldim)
    opts.dim_V =ldim(e);
    for a = 1:length(parm1)
        opts.lambda_1=parm1(a);
        for b = 1:length(parm2)
            opts.lambda_2=parm2(b);
            for i = 1:30
                [U,V,max_iter]  = CAMR(X,opts);
                [C] = Affinitysynthesis(U,V);
                W = (abs(C)+abs(C'))/2;
                [ress(max_iter,:)] = ClusteringMeasure(W,gt{1});
                res_comvsc = ress(end,:);
                disp(['l = ',  num2str(opts.dim_V), '  lambda_1 = ', num2str(opts.lambda_1), '  lambda_2 = ', num2str(opts.lambda_2),  '  ACC = ', num2str(res_comvsc(:,8))]);
            end
        end
    end
end