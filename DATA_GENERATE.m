clear all
rng(1)

%%%%%% GENERATING 2 Category output DATA %%%%%%%
num_output_categories = 2;
each_sample_size = 15;
num_true_covs = 5;
num_false_covs = 2;
num_total_covs = num_false_covs + num_true_covs;


sample_sizes = each_sample_size*ones(num_output_categories,1);

base_mean = zeros(num_true_covs,1);
for i = 1:num_true_covs
    base_mean(i) = 1+(i-1)*.1;
end
mu = repmat(base_mean,1,num_output_categories);
for i = 1:num_output_categories
    mu(:,i) = mu(:,i)*(i-1);
end
SIGMAS = cell(num_output_categories,1);
for  i =1:num_output_categories
    SIGMAS{i} = eye(num_true_covs);
end

X_mat = zeros(num_true_covs+num_false_covs,each_sample_size*num_output_categories);
for i = 1:num_output_categories
    for j = 1:each_sample_size
        X_mat(1:num_true_covs,(i-1)*each_sample_size + j) = transpose(mvnrnd(mu(:,i),SIGMAS{i},1));
    end
end
if(num_false_covs>0)
    for jj = 1:num_false_covs
        X_mat(num_true_covs+jj,:) = -1+2*rand(each_sample_size*num_output_categories,1);
    end
end

for ii = 1:num_output_categories
    if(ii == 1)
        Y_column = zeros(sample_sizes(ii),1);
    else
        Y_column = [Y_column; (ii-1)*ones(sample_sizes(ii),1)];
    end
end

Y_and_X = [Y_column, X_mat'];

random_order = randperm(size(Y_and_X,1));

Y_and_X_FINAL = Y_and_X(random_order,:);

filename = ['DATA_num_categories_',num2str(num_output_categories),'.csv'];
csvwrite(filename,Y_and_X_FINAL)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% GENERATING 3 Category output DATA %%%%%%%
num_output_categories = 3;
each_sample_size = 15;
num_true_covs = 5;
num_false_covs = 5;
num_total_covs = num_false_covs + num_true_covs;



sample_sizes = each_sample_size*ones(num_output_categories,1);

base_mean = zeros(num_true_covs,1);
for i = 1:num_true_covs
    base_mean(i) = 1+(i-1)*.1;
end
mu = repmat(base_mean,1,num_output_categories);
for i = 1:num_output_categories
    mu(:,i) = mu(:,i)*(i-1);
end
SIGMAS = cell(num_output_categories,1);
for  i =1:num_output_categories
    SIGMAS{i} = eye(num_true_covs);
end

X_mat = zeros(num_true_covs+num_false_covs,each_sample_size*num_output_categories);
for i = 1:num_output_categories
    for j = 1:each_sample_size
        X_mat(1:num_true_covs,(i-1)*each_sample_size + j) = transpose(mvnrnd(mu(:,i),SIGMAS{i},1));
    end
end
if(num_false_covs>0)
    for jj = 1:num_false_covs
        X_mat(num_true_covs+jj,:) = -1+2*rand(each_sample_size*num_output_categories,1);
    end
end

for ii = 1:num_output_categories
    if(ii == 1)
        Y_column = zeros(sample_sizes(ii),1);
    else
        Y_column = [Y_column; (ii-1)*ones(sample_sizes(ii),1)];
    end
end

Y_and_X = [Y_column, X_mat'];

random_order = randperm(size(Y_and_X,1));

Y_and_X_FINAL = Y_and_X(random_order,:);

filename = ['DATA_num_categories_',num2str(num_output_categories),'.csv'];
csvwrite(filename,Y_and_X_FINAL)

