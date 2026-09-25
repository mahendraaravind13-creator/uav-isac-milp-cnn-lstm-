%run in MATLAB by saving .m


% This is the main file and correct 12092025
% Working and can be used for dataset generation
% UAV MILP: Early stages prioritize communication (Stage 2) until fraction of R_min
% achieved, overall communication requirement enforced, sensing maximized afterward.

clc;
clear all;
close all;

%R_min = [1e8];                % total communication requirement
%p_fraction = [0.5];     % fraction of R_min to trigger sensing (early-stage rule)



% remove ii loop becasuse singlee value where ever ii is there remove instead
% one rmin value and random refraction
%

%% --------------------
% General Parameters
%% --------------------
Number_Environments = 200000;    % number of random environments to generate
master_table = table();

%% Loop over environments
for env = 1:Number_Environments

R_min = 1e7 + (3e8-1e7)*rand(1,1);

p_fraction = 0.1 + (0.9-0.1)*rand(1,1);

%J = 10;
J=15;                    % number of waypoints / stages
M = 10;                     % number of communication users (CUs)
K = 10;                     % number of sensing targets (STs)
%Lx = 500; Ly = 300;     % rectangular area (meters)
H = 70;                 % UAV altitude (meters)
T_h = 10;               % hover time at each stage (seconds)
Pt = 0.1;               % transmit power (W)
Gt = 10;                % UAV transmit antenna gain (linear)
Gc = 5;                 % CU receive antenna gain (linear)
sigma2 = 1e-9;          % noise power (W)
B_total = 10e6;         % total comm bandwidth (Hz)
B_alloc = B_total / M;  % per CU bandwidth (equal split)

% Sensing parameters
Gp = 10;             % processing gain
Gs = 5;              % ST antenna gain
sigma_rcs = 1;       % radar cross-section
lambda = 0.1;        % carrier wavelength (m)
a = 1;               % environment noise scaling

% Precompute UAV waypoints (linear interpolation)
%start_pos = [0, 0]; end_pos = [Lx, Ly];
% s = zeros(J, 2);
%for j = 1:J
%    alpha = (j-1)/(J-1);
%    s(j,:) = (1-alpha)*start_pos + alpha*end_pos;
%end

% IF called for random trajectory
J = 15;              % total slots
Lx = 500; Ly = 300;
start_pos = [0, 0]; end_pos = [Lx, Ly];
d_step = 60;   % nominal step size

s = UAV_random_trajectory(J,Lx,Ly,start_pos,end_pos,d_step);

alpha0 = (Gt * Gc * lambda^2) / (4*pi)^2;
beta0  = (Gt * Gs * sigma_rcs * lambda^2) / (4*pi)^3;

%% --------------------
% MILP parameters
%% --------------------
eta = 1e-4;        % small penalty to discourage Stage 3 slightly
epsilon = 1e-12;   % numerical epsilon



    rng('shuffle');
    fprintf('Environment %d / %d\n', env, Number_Environments);

    % Fixed positions (example)
    %CU_pos = [41.42 174.41;147.16 7.32;301.01 48.53;374.37 23.81;196.41 207.35;276.58 288.10;171.98 287.07;319.61 109.65;190.80 271.09;398.69 162.52];
    %ST_pos = [38.21 109.16;74.68 166.22;69.85 143.59;312.35 276.49;369.03 64.85;37.08 68.12;199.57 186.79;412.49 275.98;232.46 44.94;94.83 253.03];

    CU_pos = [Lx*rand(M,1), Ly*rand(M,1)];
    ST_pos = [Lx*rand(K,1), Ly*rand(K,1)];
    %% --------------------
    % Compute per-stage communication performance Psi_c_all
    %% --------------------
    Psi_c_all = zeros(J,1);
    for j = 1:J
        UAV_pos = s(j,:);
        tmp = zeros(M,1);
        for m = 1:M
            d = sqrt(H^2 + norm(UAV_pos - CU_pos(m,:))^2);
            h = alpha0 / (d^2);
            SNR = Pt * h / sigma2;
            R = B_alloc * log2(1 + SNR);
            tmp(m) = T_h * R;
        end
        Psi_c_all(j) = mean(tmp);
    end

    %% --------------------
    % Compute per-stage sensing performance Psi_s_all
    %% --------------------
    Psi_s_all = zeros(J,1);
    for j = 1:J
        sensing_stages = 1:j;
        psi_s_tmp = zeros(K,1);
        for k = 1:K
            u_k = ST_pos(k,:);
            Q_k = zeros(2,length(sensing_stages));
            inv_var = zeros(length(sensing_stages),1);
            for idx = 1:length(sensing_stages)
                jp = sensing_stages(idx);
                UAV_pos = s(jp,:);
                d = sqrt(H^2 + norm(UAV_pos - u_k)^2);
                sigma2_kjp = (a*sigma2)/(Pt*Gp*beta0) * d^4;
                inv_var(idx) = 1/(sigma2_kjp + epsilon);
                dx = (u_k(1)-UAV_pos(1))/d; dy = (u_k(2)-UAV_pos(2))/d;
                Q_k(:,idx) = [dx;dy];
            end
            J_d = diag(inv_var);
            J_u = Q_k * J_d * Q_k';
            Theta_a = J_u(1,1); Theta_b = J_u(2,2); Theta_c = J_u(1,2);
            denom = Theta_a*Theta_b - Theta_c^2;
            if denom > 0
                psi_s_tmp(k) = (Theta_a+Theta_b)/denom;
            else
                psi_s_tmp(k) = Inf;
            end
        end
        Psi_s_all(j) = mean(psi_s_tmp,'omitnan');
        if isinf(Psi_s_all(j))
            Psi_s_all(j) = 1e18;
        end
    end

    %% --------------------
    % MILP setup
    %% --------------------
    fs = 1 ./ (Psi_s_all + epsilon);  % sensing utility
    % Gamma = alphaGamma * mean(Psi_c_all);

    nvars = 3*J; intcon = 1:nvars;  % xj1,xj2,xj3
    f = zeros(nvars,1);
    for j = 1:J
        idx = (j-1)*3;
        f(idx+1) = -fs(j);    % Stage1
        f(idx+3) = -fs(j);    % Stage3
        f(idx+2) = f(idx+2) + eta; f(idx+3) = f(idx+3) + eta;
    end

    % Equality: xj1+xj2+xj3=1
    Aeq = zeros(J,nvars); beq = ones(J,1);
    for j=1:J
        idx=(j-1)*3; Aeq(j,idx+1:idx+3)=[1 1 1];
    end

    %% --------------------
    % Early-stage Stage2 enforcement
    %% --------------------
    cum_comm = 0;
    A_early=[]; b_early=[];
    for j = 1:J
        idx=(j-1)*3;
        if cum_comm < p_fraction*R_min
            % block Stage1 and Stage3
            A_row=zeros(1,nvars); A_row(idx+1)=1; A_row(idx+3)=1;
            A_early=[A_early;A_row]; b_early=[b_early;0];
        end
        cum_comm = cum_comm + Psi_c_all(j); % assume Stage2 for estimation
    end

    %% --------------------
    % Overall communication constraint
    %% --------------------
    A_comm=zeros(1,nvars);
    for j=1:J
        idx=(j-1)*3;
        A_comm(1,idx+2) = -Psi_c_all(j); % Stage2
        A_comm(1,idx+3) = -Psi_c_all(j); % Stage3
    end
    b_comm = -R_min;

    %% --------------------
    % Combine inequalities
    %% --------------------
    A = [A_comm; A_early]; b = [b_comm; b_early];
    lb=zeros(nvars,1); ub=ones(nvars,1);

    %% --------------------
    % Solve MILP
    %% --------------------
    opts = optimoptions('intlinprog','Display','off','RelativeGapTolerance',1e-4,'MaxTime',60);
    [xopt,fval,exitflag] = intlinprog(f,intcon,A,b,Aeq,beq,lb,ub,opts);
    if ~(exitflag==1||exitflag==2)
        error('MILP did not find feasible solution. exitflag=%d',exitflag);
    end

    %% --------------------
    % Decode solution
    %% --------------------
    stage_type_opt=zeros(1,J);
    for j=1:J
        idx=(j-1)*3;
        [~,pick]=max(xopt(idx+1:idx+3));
        stage_type_opt(j)=pick;
    end

    count1=sum(stage_type_opt==1); count2=sum(stage_type_opt==2); count3=sum(stage_type_opt==3);
    comm_indices=find(stage_type_opt==2|stage_type_opt==3);
    if ~isempty(comm_indices)
        achieved_avg_comm = mean(Psi_c_all(comm_indices));
    else
        achieved_avg_comm = 0;
    end
    sense_indices=find(stage_type_opt==1|stage_type_opt==3);
    if ~isempty(sense_indices)
        achieved_avg_sense = mean(Psi_s_all(sense_indices));
        achieved_avg_sensing_utility = mean(fs(sense_indices));
    else
        achieved_avg_sense=Inf;
        achieved_avg_sensing_utility=0;
    end

    aaa_rate = achieved_avg_comm;
    aaa_sensing = achieved_avg_sense;
    aaa_rate_min = p_fraction;
    aaa_stages = [count1 count2 count3];

    fprintf('Env %d: Counts -> S1=%d, S2=%d, S3=%d. ',env,count1,count2,count3);
    fprintf('Achieved avg comm=%.4g (R_min=%.4g), ',achieved_avg_comm,R_min);
    fprintf('Achieved avg sense(CRB)=%.4g, sensing utility(avg fs)=%.4g\n',achieved_avg_sense,achieved_avg_sensing_utility);

    %% --------------------
    % Build feature table
    %% --------------------
    feature_table=table;
    feature_table.X=s(:,1);
    feature_table.Y=s(:,2);

    mean_d_CU=zeros(J,1); min_d_CU=zeros(J,1); max_d_CU=zeros(J,1);
    mean_d_ST=zeros(J,1); min_d_ST=zeros(J,1); max_d_ST=zeros(J,1);

    % storage for cumulative metrics
    cumulative_comm=zeros(J,1);
    cumulative_sense=zeros(J,1);

    % running sums
    comm_sum=0;
    sense_sum=0;

    for j=1:J
        % distances
        d_cu = sqrt(H^2 + sum((CU_pos - s(j,:)).^2,2));
        mean_d_CU(j)=mean(d_cu);
        min_d_CU(j)=min(d_cu);
        max_d_CU(j)=max(d_cu);

        d_st = sqrt(H^2 + sum((ST_pos - s(j,:)).^2,2));
        mean_d_ST(j)=mean(d_st);
        min_d_ST(j)=min(d_st);
        max_d_ST(j)=max(d_st);

        % update cumulative performance (use up to j-1)
        if j>1
            comm_sum = comm_sum + Psi_c_all(j-1);
            sense_sum = sense_sum + Psi_s_all(j-1);
        end
    end

    % add features
    feature_table.MeanDistCU=mean_d_CU;
    feature_table.MinDistCU=min_d_CU;
    feature_table.MaxDistCU=max_d_CU;

    feature_table.MeanDistST=mean_d_ST;
    feature_table.MinDistST=min_d_ST;
    feature_table.MaxDistST=max_d_ST;

    feature_table.PsiComm=Psi_c_all(:);
    feature_table.PsiSense=Psi_s_all(:);

    % NEW: environment constants (same for all stages in one env)
    feature_table.Rmin=R_min*ones(J,1);
    feature_table.FractionP=p_fraction*ones(J,1);

    % label from optimization (MILP solution)
    feature_table.Label=stage_type_opt(:);

    % append to master dataset
    master_table=[master_table; feature_table];

end

%% --------------------
% Save results
%% --------------------
save('master_table.mat','master_table');
writetable(master_table,'master_table.csv');
fprintf('Done. Master table saved as master_table.mat and master_table.csv\n');