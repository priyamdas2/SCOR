clear all
rng(1)
num_output_categories = 3;  % Input 2 or 3 depending on which data to work on
which_method = 3;           % 1 = ULBA, 2 = EHUM, 3 = SHUM
execution_time = 3600;      % Max time allowed (in secs) for the optimization
use_parallel = 1;           % 0 = No parallel, 1 = parallel

%%% DATA READING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

filename = ['DATA_num_categories_',num2str(num_output_categories),'.csv'];
DATA_scrambled = csvread(filename);


%%%% Re-structuring data before input in likelihood function readable form
sample_sizes = zeros(num_output_categories,1);
for ii = 1:num_output_categories
    if(ii == 1)
        which_positions = find(DATA_scrambled(:,1) == (ii-1));
        Location_number_list = which_positions;
        sample_sizes(ii) = max(size(which_positions));
    else
        which_positions = find(DATA_scrambled(:,1) == (ii-1));
        Location_number_list = [Location_number_list; which_positions];
        sample_sizes(ii) = max(size(which_positions));
    end
end
% 'Location_number_list' gives the positions so the categories are ordered.

DATA_organized = DATA_scrambled(Location_number_list,:);
X_mat_temp = DATA_organized';
X_mat = X_mat_temp;
X_mat(1,:) = [];

%%%% Deciding which method to run %%%%%%%%%%%%%%%%%%%%%%%%%%

if(which_method == 1)
    fun = @(theta)-fun_ULBA(theta, sample_sizes, X_mat);
else
    if(which_method == 2)
        fun = @(theta)-fun_EHUM_efficient(theta, sample_sizes, X_mat);
    else
        if(which_method == 3)
            fun = @(theta)-fun_SSHUM(theta, sample_sizes, X_mat);
        end
    end
end

method_names = ['Computing ULBA...';'Computing EHUM...';'Computing SHUM...'];
if(which_method == 1)               % Prints out which method has been run
    disp(method_names(1,:))
else
    if(which_method == 2)
        disp(method_names(2,:))
    else
        if(which_method == 3)
            disp(method_names(3,:))
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%RMPSSP Algorithm%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M = size(X_mat,1);                 % number of parameters on simplex
no_loops = 1000;                   % max_runs
maximum_iteration = 10000;         % max_iters
epsilon_start = 2;                 % initial global step-size
rho_1 = 2;                         % step decay rate for 1st loop
rho_2 = 2;                         % step decay rate for 2nd loop onwards
tol_fun = 10^-6;                   % tol_fun
tol_fun_2 = 10^-3;                 % tol_fun_2
epsilon_cut_off = 10^(-20);        % lower bound of global step-size
theta_cut_off = 10^(-6);           % sparsity threshold

%%%%%%%%%%%%%%%%%%%%%%%
%%%% Randomly generated staring point

starting_point = -1 + 2*rand(M,1);
starting_point = starting_point/norm(starting_point);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;

array_of_values = zeros(maximum_iteration,1);
theta_array = zeros(no_loops, M);
Loop_solution = zeros(no_loops, 1);

last_toc = 0;
for iii = 1:no_loops
    epsilon = epsilon_start;
    epsilon_decreasing_factor = rho_2;
    if(iii == 1)
        epsilon_decreasing_factor = rho_1;
        theta = starting_point;
    else
        theta = transpose(theta_array((iii-1),:));
    end
    M = max(size(theta,1),size(theta,2));
    
    
    for i = 1:maximum_iteration
        if(toc>execution_time)
            break
        end
        current_lh = fun(theta);
        
        %%%% Time display %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        toc_now = toc;
        if(toc_now - last_toc > 2)
            if(rem(round(toc_now),5) == 0)
                disp(-current_lh);
                last_toc = toc_now;
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % [iii,i/1000,log10(epsilon), transpose(theta),  current_lh,norm(theta)]
        
        
        total_lh = zeros(2*M,1);
        matrix_update_at_h = zeros(M,2*M);
        
        total_lh_alt = zeros(2*M,1);
        matrix_update_at_h_alt = zeros(M,2*M);
        
        if(use_parallel == 0)
            for location_number = 1:(2*M)
                change_loc  = ceil(location_number/2);
                possibility = theta;
                %significant_positions = [1:M];
                %                   %significant_positions(change_loc) = [];
                possibility(change_loc) = 0;
                significant_positions = find(gt(abs(possibility), theta_cut_off*ones(M,1)));
                possibility = zeros(M,1);
                possibility_alt = zeros(M,1);
                epsilon_temp = ((-1)^location_number)*epsilon;
                M_here = length(significant_positions)+1;
                if(M_here >= 2)
                    D = (2*sum(theta(significant_positions)))^2 - 4*(M_here-1)*...
                        (2*theta(change_loc)*epsilon_temp+epsilon_temp^2);
                    while(D <0 && abs(epsilon_temp) > epsilon_cut_off)
                        epsilon_temp = epsilon_temp/epsilon_decreasing_factor;
                        D = (2*sum(theta(significant_positions)))^2 - 4*(M_here-1)*...
                            (2*theta(change_loc)*epsilon_temp+epsilon_temp^2);
                    end
                    if(D >= 0)
                        possibility(change_loc) = theta(change_loc) + epsilon_temp;
                        possibility_alt(change_loc) = theta(change_loc) + epsilon_temp;
                        a = M_here-1;
                        b = 2*sum(theta(significant_positions));
                        c = 2*theta(change_loc)*epsilon_temp+epsilon_temp^2;
                        D = b^2 - 4*a*c;
                        t_here = (-b + sqrt(D))/(2*a);
                        t_here_alt = (-b - sqrt(D))/(2*a);
                        possibility(significant_positions) = theta(significant_positions) + t_here;
                        possibility_alt(significant_positions) = theta(significant_positions) + t_here_alt;
                        total_lh(location_number) = fun(possibility);
                        total_lh_alt(location_number) = fun(possibility_alt);
                    else
                        possibility = theta;
                        possibility_alt = theta;
                        total_lh(location_number) = current_lh;
                        total_lh_alt(location_number) = current_lh;
                    end
                else
                    possibility(change_loc) = round(theta(change_loc));
                    possibility_alt(change_loc) = round(theta(change_loc));
                    total_lh(location_number) = fun(possibility);
                    total_lh_alt(location_number) = fun(possibility_alt);
                end
                
                matrix_update_at_h(:,location_number) = possibility;
                matrix_update_at_h_alt(:,location_number) = possibility_alt;
            end
        else
            parfor location_number = 1:(2*M)
                change_loc  = ceil(location_number/2);
                possibility = theta;
                %significant_positions = [1:M];
                %                   %significant_positions(change_loc) = [];
                possibility(change_loc) = 0;
                significant_positions = find(gt(abs(possibility), theta_cut_off*ones(M,1)));
                possibility = zeros(M,1);
                possibility_alt = zeros(M,1);
                epsilon_temp = ((-1)^location_number)*epsilon;
                M_here = length(significant_positions)+1;
                if(M_here >= 2)
                    D = (2*sum(theta(significant_positions)))^2 - 4*(M_here-1)*...
                        (2*theta(change_loc)*epsilon_temp+epsilon_temp^2);
                    while(D <0 && abs(epsilon_temp) > epsilon_cut_off)
                        epsilon_temp = epsilon_temp/epsilon_decreasing_factor;
                        D = (2*sum(theta(significant_positions)))^2 - 4*(M_here-1)*...
                            (2*theta(change_loc)*epsilon_temp+epsilon_temp^2);
                    end
                    if(D >= 0)
                        possibility(change_loc) = theta(change_loc) + epsilon_temp;
                        possibility_alt(change_loc) = theta(change_loc) + epsilon_temp;
                        a = M_here-1;
                        b = 2*sum(theta(significant_positions));
                        c = 2*theta(change_loc)*epsilon_temp+epsilon_temp^2;
                        D = b^2 - 4*a*c;
                        t_here = (-b + sqrt(D))/(2*a);
                        t_here_alt = (-b - sqrt(D))/(2*a);
                        possibility(significant_positions) = theta(significant_positions) + t_here;
                        possibility_alt(significant_positions) = theta(significant_positions) + t_here_alt;
                        total_lh(location_number) = fun(possibility);
                        total_lh_alt(location_number) = fun(possibility_alt);
                    else
                        possibility = theta;
                        possibility_alt = theta;
                        total_lh(location_number) = current_lh;
                        total_lh_alt(location_number) = current_lh;
                    end
                else
                    possibility(change_loc) = round(theta(change_loc));
                    possibility_alt(change_loc) = round(theta(change_loc));
                    total_lh(location_number) = fun(possibility);
                    total_lh_alt(location_number) = fun(possibility_alt);
                end
                
                matrix_update_at_h(:,location_number) = possibility;
                matrix_update_at_h_alt(:,location_number) = possibility_alt;
            end
        end
        
        
        [M_root,I_root] = min(total_lh);
        [M_root_alt,I_root_alt] = min(total_lh_alt);
        
        if(M_root <M_root_alt)
            if(M_root < current_lh)
                theta = matrix_update_at_h(:,I_root);
            end
            final_value =  min(M_root,current_lh);
        else
            if(M_root_alt < current_lh)
                theta = matrix_update_at_h_alt(:,I_root_alt);
            end
            final_value =  min(M_root_alt,current_lh);
        end
        
        array_of_values(i) =  min(M_root,current_lh);
        
        if(i > 1)
            if(abs(array_of_values(i) - array_of_values(i-1)) < tol_fun)
                if(epsilon > epsilon_decreasing_factor*epsilon_cut_off)
                    epsilon = epsilon/epsilon_decreasing_factor;
                else
                    break
                end
            end
        end
        
    end
    
    theta_array(iii,:) = transpose(theta);
    Loop_solution(iii) = fun(theta);
    transpose(theta);
    if(iii > 1)
        old_soln = theta_array(iii-1,:);
        new_soln = theta_array(iii,:);
        if(norm(old_soln - new_soln) <tol_fun_2)
            break
        end
    end
end

answer = -fun(theta);
required_time = toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



theta                               % Estimated coefficient vector
answer                              % Value of ULBA/EHUM/SHUM
required_time                       % Computation time




