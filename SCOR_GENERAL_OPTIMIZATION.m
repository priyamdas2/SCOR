clear all
rng(1)
% Choice of function or user can also define his own function
which_function = 1;   % 1 = Negative log of product, 2 = Griewank
                      % 3 = Negative Sum Squares Function,
                      % 4 = Exponential Function, 5 = Easom,
                      % 6 = Rastrigin, 7 = Ackley.
use_parallel = 0;           % 0 = No parallel, 1 = parallel                     
M = 5;                      % dimension of the BETA vector
execution_time = 3600;      % Max time allowed (in secs) for the optimization

%%%%%%%% parameters %%%%%%%%%%%%
no_loops = 1000;                   % max_runs
maximum_iteration = 10000;         % max_iters
epsilon_start = 2;                 % initial global step-size
rho_1 = 2;                         % step decay rate for 1st loop
rho_2 = 2;                         % step decay rate for 2nd loop onwards
adjustment_factor = 10^20;         % adjustment_factor
tol_fun = 10^-6;                   % tol_fun
tol_fun_2 = 10^-20;                % tol_fun_2
epsilon_cut_off = 10^(-20);        % lower bound of global step-size
theta_cut_off = 10^(-6);           % sparsity threshold
array_of_values = zeros(maximum_iteration,1);


%%%%%% BENCHMARK OBJECTIVE FUNCTIONS %%%%%%%

% Negative log of product of absolute values
if(which_function == 1)
    fun = @(x)-(M/2)*log(M)-sum(log(abs(x)));
    fun_ga = fun;
    fun_sa = @(x) -sum(log(abs(x)));
    soln = (1/sqrt(M))*ones(M,1);
end

% Griewank
if(which_function == 2)
    fun = @(x) ((1/4000)*sum((x-1/sqrt(M)).^2) - prod(cos(transpose((x-1/sqrt(M)))./sqrt(1:M)))+1);
    soln = (1/sqrt(M))*ones(M,1);
end


% Negative Sum Squares Function
if(which_function == 3)
    fun = @(x)M-sum([1:M]*(x.^2));
    soln = [zeros(M-1,1);1];
end

% Exponential Function
if(which_function == 4)
    upto_M = M-1;
    fun = @(x) 1-exp(-0.5*sum(x(1:upto_M).^2));
    soln = [zeros(M-1,1);1];
end

% Easom
if(which_function == 5)
    upto_M = 2;
    fun = @(x) 1-prod(cos(sqrt(upto_M)*pi*(x(1:upto_M))))*exp(-sum((x(1:upto_M)-sqrt(1/upto_M)).^2));
    soln = [(1/sqrt(upto_M))*ones(M,1);zeros((M-upto_M),1)];
end

% Rastrigin
if(which_function == 6)
    
    fun = @(x) (10*M+sum((x+1/sqrt(M)).^2-10*cos(2*pi*(x+1/sqrt(M)))));
    soln = -(1/sqrt(M))*ones(M,1);
end

% Ackley
if(which_function == 7)
    fun = @(x) (-20*exp(-0.2*sqrt(1/M*sum((x-1/sqrt(M)).^2)))-exp(1/M*sum(cos(2*pi*(x-1/sqrt(M)))))+20+exp(1));
    soln = (1/sqrt(M))*ones(M,1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Initial values %%%%%%%%
starting_point = -1 + 2*rand(M,1);
starting_point = starting_point/norm(starting_point);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%% SCOR Code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
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
        %[iii,i/1000,log10(epsilon), current_lh,norm(theta)]
        
        
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
        old_soln = round(theta_array(iii-1,:)*(adjustment_factor))/adjustment_factor;
        new_soln = round(theta_array(iii,:)*(adjustment_factor))/adjustment_factor;
        if(norm(old_soln - new_soln) <tol_fun_2)
            break
        end
    end
end

initial_obj_value = fun(starting_point/norm(starting_point));
theta_final = theta/norm(theta);
answer = fun(theta_final);
true_minimum = fun(soln);
required_time = toc;

theta_final    % Estimated global minimum solution point
answer         % Estimated global minimum achieved by SCOR
true_minimum   % TRUE global minimum
required_time

