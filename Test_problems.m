clear all,
close all,

% Add path to support functions
addpath(genpath('./support_fun'))

% Add path to the Oseledets TT-Toolbox, the public code for SSCG and the
% htucker toolbox for tensor-times-matrix product, the matricization and
% the dematricization
% (before running this code, ensure to download the Oseledets TT-Toolbox, 
% the public code for SSCG, and htucker toolbox and place them in the 
% same directory as this script)
addpath(genpath('./oseledets_TT-Toolbox'))
addpath(genpath('./pub_sscg'))
addpath(genpath('./htucker_1.2'))

format shorte
format compact
rng(12345);

% Number of modes of the problem
d = 3;

% Choose the dimension along each mode
n = 500;

h = 1 / (n+1);

% Fixing the nodes to discretize the problem
x_nodes = linspace(0, 1, n+3)'; x=x_nodes;
y_nodes = linspace(0, 1, n+3)'; y=y_nodes;
z_nodes = linspace(0, 1, n+3)'; z=z_nodes;

% Generating useful quantities such as the identity matrix
% and the matrix R for the discretization of the first derivative 
e = ones(n+1, 1);
R=sparse([1*diag(ones(n+1,1))-diag(ones(n,1),1),[zeros(n,1);-1]]);
I = speye(n+1, n+1); 

% Choose the problem:
iproblem=input('problem? (1/2/3/4) ');

switch iproblem

case 1 % Problem 1 
% Laplacian operator
a1 = ones(n+1, 1); a1 = a1 / norm(a1);
b1 = eye(n+1, 1); b1 = b1 / norm(b1);
c1 = eye(n+1, 1); c1 = c1 / norm(c1);
A{1,1} = (R*R');  A{1,2} = I;   A{1,3} = I;
A{2,1} = I;  A{2,2} = (R*R');    A{2,3} = I;
A{3,1} = I;   A{3,2} = I ;  A{3,3} = (R*R');

% Fix the convergence tolerance
options.tol = 1e-4; 

case 2
% Generic elliptic operator
%  L(u) = ( (x+1)(y+1) u_x)_x + ( (x+1)(y+1) u_y)_y + ((1+x)(y+1) u_z)_z
a1 = ones(n+1, 1); a1 = a1 / norm(a1);
b1 = eye(n+1, 1); b1 = b1 / norm(b1);
c1 = eye(n+1, 1); c1 = c1 / norm(c1);
A{1,1} = (R*spdiags((2+x(1:n+2)+x(2:n+3))/2,0,n+2,n+2)*R');  
A{1,2} = spdiags(y+1,0,n+1,n+1);   A{1,3} = I; 
A{2,1} = spdiags(1+x,0,n+1,n+1);  
A{2,2} = (R*spdiags((2+y(1:n+2)+y(2:n+3))/2,0,n+2,n+2)*R'); A{2,3} = I; 
A{3,1} = spdiags(1+x(2:n+2),0,n+1,n+1);   A{3,2} = spdiags(1+y(2:n+2),0,n+1,n+1);  
A{3,3} = (R*R');
 
 % Fix the convergence tolerance
 options.tol = 1e-4; 


case 3
% Diffusion-reaction operator
%   L(u)=mu  Delta u + 1000 (x+1)(z+1) u
a1 = ones(n+1, 1); a1 = a1 / norm(a1);
b1 = eye(n+1, 1); b1 = b1 / norm(b1);
c1 = eye(n+1, 1); c1 = c1 / norm(c1);
mu=0.1*n^2;
A{1,1} = mu*(R*R');  A{1,2} = I;   A{1,3} = I;
A{2,1} = I;  A{2,2} = mu*(R*R');    A{2,3} = I;
A{3,1} = I;   A{3,2} = I ;  A{3,3} = mu*(R*R');
A{4,1} = spdiags(1000*(x(2:n+2)+1),0,n+1,n+1); 
A{4,2} = I; A{4,3} = spdiags((z(2:n+2)+1),0,n+1,n+1);

% Fix the convergence tolerance
options.tol = 1e-4; 


case 4
% Generic elliptic operator
%  L(u) = ( exp(-x)exp(-y) u_x)_x + ( exp(x)exp(y) u_y)_y + ((1+x)(y+1) u_z)_z
a1 = ones(n+1, 1); a1 = a1 / norm(a1);
b1 = eye(n+1, 1); b1 = b1 / norm(b1);
c1 = eye(n+1, 1); c1 = c1 / norm(c1);
xnew=10*ones(1,n+2);
n4=floor((n+2)/4);
xnew(n4:2*n4)=0.01*xnew(n4:2*n4);
A{1,1} = (R*spdiags(xnew,0,n+2,n+2)*R');
A{1,2} = I;   A{1,3} = I;
A{2,1} = I;
A{2,2} = (R*spdiags(xnew,0,n+2,n+2)*R'); A{2,3} = I;
A{3,1} = I;   A{3,2} = I;
A{3,3} = (R*spdiags(xnew,0,n+2,n+2)*R');
 
% Fix the convergence tolerance
options.tol = 1e-4;

otherwise
   disp('error data')
   return

end

% Building the TT representation of the operator and the right-hand side
% (AMEn)
Ctt = tt_tensor({a1, b1, c1});
Ctt = Ctt*(1/norm(Ctt));

% Building the operator A0 in TT format (AMEn)
[nterms,d]=size(A);
I=speye(n+1);

A0 = tt_matrix({full(A{1,1}); full(A{1, 2}); full(A{1,3})});
for kk = 2:nterms
    A0 = A0 + tt_matrix({full(A{kk,1}); full(A{kk, 2}); full(A{kk,3})});
end

% Building Tucker right-hand side
C.core = 1;
C.factors{1,1} = a1;
C.factors{1,2} = b1;
C.factors{1,3} = c1;

% Solver parameter
options.maxit = 80; % Maximum number of iterations
options.tol_tr = 1e-16; % Truncation tolerance
options.X0 = 0; % Initial guess
options.innsolver = 2; % Inner solver for alpha (and beta)
options.single_flag = true;
m = 10; 
options.maxrank = m; % Maximum rank

%--------------------------------------------------------------
% RUNING Tk-SS-SD
% Parameters for inner solvers
% innerouter: true = preconditioning through inner-outer strategy; false = otherwise.
options.innerouter = false; 

disp('Tucker SSSD + SSSD')

tic; 
[X_tucker_ss_sd, info_tucker_ss_sd] = tucker_sssd_pc(A, C, [], options); 
t_ss_sd =toc;
it_ss_sd = info_tucker_ss_sd{3,1}; 

if n> 500
    Xcore = X_tucker_ss_sd.core;
    Xfact = X_tucker_ss_sd.factors;
    Res_core = C.core;
    Res_fact = C.factors;
    for l = 1:nterms
       Res_core = tucker_blkdiag((Res_core), -Xcore);
       for k = 1:d  
            Res_fact{k} = [Res_fact{k}, A{l, k}*Xfact{k}];
            [Ures, Sres, Vres] = svd(Res_fact{k}, 'econ');
            sres = diag(Sres)./sum(diag(Sres));
            rres = (sum(sres > 1e-16));
            Res_fact{k} = Ures(:, 1:rres);
            Res_core = tucker_times_matrix(Res_core, (Sres(1:rres, 1:rres)*(Vres(:, 1:rres)')), k);
       end
    end
    res_norm_ss_sd = norm(Res_core, 'fro');
    clear Xcore Xfact
    clear Res_core Res_fact
else
    res_norm_ss_sd = norm((tucker_times_matrix(C.core, C.factors)) - ...
        tucker_AX(A, X_tucker_ss_sd), 'fro')/norm(tucker_times_matrix(C.core, C.factors), 'fro');
end
fprintf('Tk-SS-SD\t %.4e\t %.4e \n',  res_norm_ss_sd, t_ss_sd)
disp('----------------------------------------------------------------')

%--------------------------------------------------------------

% Tk-SS_CG
% Parameters for inner solvers
options.innsolver = 2;
options.innerouter = 0; % 1 = preconditioning through inner-outer strategy; 0 = otherwise.

disp('Tucker SSCG + SSSD')

tic; 
[X_tucker_ss_cg, info_tucker_ss_cg] = tucker_sscg_pc(A, C, [], options); 
t_ss_cg =toc;
it_ss_cg = info_tucker_ss_cg{3,1}; 

if n> 500
    Xcore = X_tucker_ss_cg.core;
    Xfact = X_tucker_ss_cg.factors;
    Res_core = C.core;
    Res_fact = C.factors;
    for l = 1:nterms
       Res_core = tucker_blkdiag((Res_core), -Xcore);
       for k = 1:d  
            Res_fact{k} = [Res_fact{k}, A{l, k}*Xfact{k}];
            [Ures, Sres, Vres] = svd(Res_fact{k}, 'econ');
            sres = diag(Sres)./sum(diag(Sres));
            rres = (sum(sres > 1e-16));
            Res_fact{k} = Ures(:, 1:rres);
            Res_core = tucker_times_matrix(Res_core, (Sres(1:rres, 1:rres)*(Vres(:, 1:rres)')), k);
       end
    end
    res_norm_ss_cg = norm(Res_core, 'fro');
    clear Xcore Xfact
    clear Res_core Res_fact
else
    res_norm_ss_cg = norm((tucker_times_matrix(C.core, C.factors)) - ...
    tucker_AX(A, X_tucker_ss_cg), 'fro')/norm(tucker_times_matrix(C.core, C.factors), 'fro');
end
fprintf('Tk-SS-CG \t %.4e\t %.4e \n',  res_norm_ss_cg, t_ss_cg)
disp('----------------------------------------------------------------')


disp('----------------------------------------------------------------')

%--------------------------------------------------------------

% Building the preconditioner using EIG function
fprintf('Computing the preconditioner\n')
clearvars M

tic;
q = 1;
Lap = -(R*R');
%[eigVECLap1,eigVALLap1] = eig(-full(Lap));
[eigVECLap1, eigVALLap1] = eigs(-(Lap), n+1);
M.E = expsum_preconditioner(q, repmat({diag(eigVALLap1)}, 1, d),... 
    repmat({eigVECLap1}, 1, d));
eta = pi/sqrt(q);
t = exp(eta*(-q:q));
M.c = eta*t;
t_Peig = toc;
clearvars tmpE E
clearvars eigVECLap1 eigVALLap1
fprintf("Prec-EIG in %.4e\n", t_Peig)

%--------------------------------------------------------------

% P-EIGs + Tk-SS-SD
% Parameters for the preconditioner
M.fftprec = false; % True = preconditioning through FFT; False = EIGs.
options.innsolver = 2;

fprintf('PREC Tucker-SSSD + SSSD \n')

tic; 
[X_Peig_tucker_ss_sd, info_Peig_tucker_ss_sd] = tucker_sssd_pc(A, C, M, options); 
t_Peig_ss_sd =toc;
it_Peig_ss_sd = info_Peig_tucker_ss_sd{3,1};

if n > 500
    Xcore = X_Peig_tucker_ss_sd.core;
    Xfact = X_Peig_tucker_ss_sd.factors;
    Res_core = C.core;
    Res_fact = C.factors;
    for l = 1:nterms
       Res_core = tucker_blkdiag((Res_core), -Xcore);
       for k = 1:d  
            Res_fact{k} = [Res_fact{k}, A{l, k}*Xfact{k}];
            [Ures, Sres, Vres] = svd(Res_fact{k}, 'econ');
            sres = diag(Sres)./sum(diag(Sres));
            rres = (sum(sres > 1e-16));
            Res_fact{k} = Ures(:, 1:rres);
            Res_core = tucker_times_matrix(Res_core, (Sres(1:rres, 1:rres)*(Vres(:, 1:rres)')), k);
       end
    end
    res_norm_Peig_ss_sd = norm(Res_core, 'fro');
    clear Xcore Xfact
    clear Res_core Res_fact

else
    res_norm_Peig_ss_sd = norm((tucker_times_matrix(C.core, C.factors)) - ...
    tucker_AX(A, X_Peig_tucker_ss_sd), 'fro')/norm(tucker_times_matrix(C.core, C.factors), 'fro');
end
fprintf('P-EIG Tk-SS-SD\t %.4e\t %.4e\n',  ...
res_norm_Peig_ss_sd, t_Peig_ss_sd)
fprintf('time P-EIG + P-EIG Tk-SS-SD\t %.4e \n',  ...
t_Peig_ss_sd + t_Peig)
disp('----------------------------------------------------------------')

%--------------------------------------------------------------

% P-EIGs + Tk-SS-CG
% Parameters for the preconditioner
M.fftprec = false; % True = preconditioning through FFT; False = EIGs.
options.innsolver = 2;

fprintf('PREC Tucker-SSCG + SSSD \n')

tic; 
[X_Peig_tucker_ss_cg, info_Peig_tucker_ss_cg] = tucker_sscg_pc(A, C, M, options); 
t_Peig_ss_cg = toc;
it_Peig_ss_cg = info_Peig_tucker_ss_cg{3,1};

if n > 500
    Xcore = X_Peig_tucker_ss_cg.core;
    Xfact = X_Peig_tucker_ss_cg.factors;
    Res_core = C.core;
    Res_fact = C.factors;
    for l = 1:nterms
       Res_core = tucker_blkdiag((Res_core), -Xcore);
       for k = 1:d  
            Res_fact{k} = [Res_fact{k}, A{l, k}*Xfact{k}];
            [Ures, Sres, Vres] = svd(Res_fact{k}, 'econ');
            sres = diag(Sres)./sum(diag(Sres));
            rres = (sum(sres > 1e-16));
            Res_fact{k} = Ures(:, 1:rres);
            Res_core = tucker_times_matrix(Res_core, (Sres(1:rres, 1:rres)*(Vres(:, 1:rres)')), k);
       end
    end
    res_norm_Peig_ss_cg = norm(Res_core, 'fro');
    clear Xcore Xfact
    clear Res_core Res_fact

else
    res_norm_Peig_ss_cg = norm((tucker_times_matrix(C.core, C.factors)) - ...
    tucker_AX(A, X_Peig_tucker_ss_cg), 'fro')/norm(tucker_times_matrix(C.core, C.factors), 'fro');
end
fprintf('P-EIG Tk-SS-CG\t %.4e\t %.4e\n',  ...
res_norm_Peig_ss_cg, t_Peig_ss_cg)
fprintf('time P-EIG + P-EIG Tk-SS-CG\t %.4e\n',  ...
t_Peig + t_Peig_ss_cg)
disp('----------------------------------------------------------------')

%--------------------------------------------------------------

% Building preconditioner FFT (DST-I)
fprintf('Computing exp prec via DST-I')
clearvars M

tic
eta = pi/sqrt(q);
t = exp(eta*(-q:q));
N = n+1;
j = (1:N)';
lambda = 2 - 2 * cos(j * pi / (N + 1));
M.filter = cell(1, 2*q+1);
for jj = 1 : 2*q + 1
    M.filter{jj} = exp(-t(jj) * lambda);
end
M.c = eta*t;
t_PFTT = toc;

fprintf("prec in %e\n", t_PFTT)

%--------------------------------------------------------------

% P-FTT + Tk-SS-SD
options.innsolver = 2;
M.fftprec = true; % FFT preconditioner
fprintf('PREC FFT Tk-SS-SD\n')

tic; 
[X_PFTT_tucker_ss_sd, info_PFTT_tucker_ss_sd] = tucker_sssd_pc(A, C, M, options); 
t_PFTT_ss_sd =toc;
it_PFTT_ss_sd = info_PFTT_tucker_ss_sd{3,1};

if n > 500
    Xcore = X_PFTT_tucker_ss_sd.core;
    Xfact = X_PFTT_tucker_ss_sd.factors;
    Res_core = C.core;
    Res_fact = C.factors;
    for l = 1:nterms
       Res_core = tucker_blkdiag((Res_core), -Xcore);
       for k = 1:d  
            Res_fact{k} = [Res_fact{k}, A{l, k}*Xfact{k}];
            [Ures, Sres, Vres] = svd(Res_fact{k}, 'econ');
            sres = diag(Sres)./sum(diag(Sres));
            rres = (sum(sres > 1e-16));
            Res_fact{k} = Ures(:, 1:rres);
            Res_core = tucker_times_matrix(Res_core, (Sres(1:rres, 1:rres)*(Vres(:, 1:rres)')), k);
       end
    end
    res_norm_PFTT_ss_sd = norm(Res_core, 'fro');
    clear Xcore Xfact
    clear Res_core Res_fact
else
    res_norm_PFTT_ss_sd = norm((tucker_times_matrix(C.core, C.factors)) - ...
    tucker_AX(A, X_PFTT_tucker_ss_sd), 'fro')/norm(tucker_times_matrix(C.core, C.factors), 'fro');
end
fprintf('P-FTT Tk-SS-SD \t %.4e\t %.4e \n',  ...
    res_norm_PFTT_ss_sd, t_PFTT_ss_sd)
fprintf('time P-FTT + P-FTT Tk-SS-SD\t %.4e\n',  ...
t_PFTT_ss_sd + t_PFTT)


disp('----------------------------------------------------------------')

%--------------------------------------------------------------

% P-FTT + Tk-SS-SD

options.innsolver = 2;
M.fftprec = true; % FFT preconditioner

fprintf('PREC FFT Tucker-SS-CG + SSSD\n')
tic; 
[X_PFTT_tucker_ss_cg, info_PFTT_tucker_ss_cg] = tucker_sscg_pc(A, C, M, options); 
t_PFTT_ss_cg = toc;
it_PFTT_ss_cg = info_PFTT_tucker_ss_cg{3,1};

if n > 500
    Xcore = X_PFTT_tucker_ss_cg.core;
    Xfact = X_PFTT_tucker_ss_cg.factors;
    Res_core = C.core;
    Res_fact = C.factors;
    for l = 1:nterms
       Res_core = tucker_blkdiag((Res_core), -Xcore);
       for k = 1:d  
            Res_fact{k} = [Res_fact{k}, A{l, k}*Xfact{k}];
            [Ures, Sres, Vres] = svd(Res_fact{k}, 'econ');
            sres = diag(Sres)./sum(diag(Sres));
            rres = (sum(sres > 1e-16));
            Res_fact{k} = Ures(:, 1:rres);
            Res_core = tucker_times_matrix(Res_core, (Sres(1:rres, 1:rres)*(Vres(:, 1:rres)')), k);
       end
    end
    res_norm_PFTT_ss_cg = norm(Res_core, 'fro');
    clear Xcore Xfact
    clear Res_core Res_fact
else
    res_norm_PFTT_ss_cg = norm((tucker_times_matrix(C.core, C.factors)) - ...
    tucker_AX(A, X_PFTT_tucker_ss_cg), 'fro')/norm(tucker_times_matrix(C.core, C.factors), 'fro');
end
fprintf('P-FFT Tk-SS-CG sscg\t %.4e\t %.4e \n',  ...
    res_norm_PFTT_ss_cg, t_PFTT_ss_cg)
fprintf('time P-FFT + P-FFT Tk-SS-CG\t %.4e\n',  ...
t_PFTT_ss_cg + t_PFTT)

disp('----------------------------------------------------------------')

%--------------------------------------------------------------

%% Innerouter precond Tk-SS-SD
options.innerouter = 1; % Inner-outer strategy for preconditioning (1 = inner-outer; 0 = otherwise).

fprintf('PREC Tucker-SS-SD + SSCG w/ innerouter\n')
tic; 
[X_Pio_tucker_ss_sd, info_Pio_tucker_ss_sd] = tucker_sssd_pc(A, C, [], options); 
t_Pio_ss_sd =toc;
it_Pio_ss_sd =  info_Pio_tucker_ss_sd{3,1};

if n > 500
    Xcore = X_Pio_tucker_ss_sd.core;
    Xfact = X_Pio_tucker_ss_sd.factors;
    Res_core = C.core;
    Res_fact = C.factors;
    for l = 1:nterms
       Res_core = tucker_blkdiag((Res_core), -Xcore);
       for k = 1:d  
            Res_fact{k} = [Res_fact{k}, A{l, k}*Xfact{k}];
            [Ures, Sres, Vres] = svd(Res_fact{k}, 'econ');
            sres = diag(Sres)./sum(diag(Sres));
            rres = (sum(sres > 1e-16));
            Res_fact{k} = Ures(:, 1:rres);
            Res_core = tucker_times_matrix(Res_core, (Sres(1:rres, 1:rres)*(Vres(:, 1:rres)')), k);
       end
    end
    res_norm_Pio_ss_sd = norm(Res_core, 'fro');
    clear Xcore Xfact
    clear Res_core Res_fact
else
    res_norm_Pio_ss_sd = norm((tucker_times_matrix(C.core, C.factors)) - ...
    tucker_AX(A, X_Pio_tucker_ss_sd), 'fro')/norm(tucker_times_matrix(C.core, C.factors), 'fro');
end
fprintf('TuckerSsCG_PC sscg - innerouter\t %.4e\t %.4e \n',  ...
    res_norm_Pio_ss_sd, t_Pio_ss_sd)

disp('----------------------------------------------------------------')

%--------------------------------------------------------------

% Innerouter precond Tk-SS-CG

options.innerouter = 1;

fprintf('PREC Tk-SS-CG + SSSD w/ innerouter\n')

tic; 
[X_Pio_tucker_ss_cg, info_Pio_tucker_ss_cg] = tucker_sscg_pc(A, C, [], options); 
t_Pio_ss_cg =toc;
it_Pio_ss_cg =  info_Pio_tucker_ss_cg{3,1};

if n > 500
    Xcore = X_Pio_tucker_ss_cg.core;
    Xfact = X_Pio_tucker_ss_cg.factors;
    Res_core = C.core;
    Res_fact = C.factors;
    for l = 1:nterms
       Res_core = tucker_blkdiag((Res_core), -Xcore);
       for k = 1:d  
            Res_fact{k} = [Res_fact{k}, A{l, k}*Xfact{k}];
            [Ures, Sres, Vres] = svd(Res_fact{k}, 'econ');
            sres = diag(Sres)./sum(diag(Sres));
            rres = (sum(sres > 1e-16));
            Res_fact{k} = Ures(:, 1:rres);
            Res_core = tucker_times_matrix(Res_core, (Sres(1:rres, 1:rres)*(Vres(:, 1:rres)')), k);
       end
    end
    res_norm_Pio_ss_cg = norm(Res_core, 'fro');
    clear Xcore Xfact
    clear Res_core Res_fact
else
    res_norm_Pio_ss_cg = norm((tucker_times_matrix(C.core, C.factors)) - ...
    tucker_AX(A, X_Pio_tucker_ss_cg), 'fro')/norm(tucker_times_matrix(C.core, C.factors), 'fro');
end
fprintf('P-innerouter Tk-SS-SG\t %.4e\t %.4e \n',  ...
    res_norm_Pio_ss_cg, t_Pio_ss_cg)

%--------------------------------------------------------------

% AMEn

fprintf('----------AMEn Comparison-----------------\n')
maxrank = [1, options.maxrank, options.maxrank, 1];
tol = options.tol; maxit = options.maxit; tol_tr = options.tol_tr;

tic;
[x,testdata,z] = amen_solve2(A0, Ctt, tol, 'x0', 0*Ctt, ...
    'rmax', max(maxrank),'tol_exit', tol, ...
    'trunc_norm', 'resid');%,'local_prec', 'ljacobi');
t_amen = toc;
res_amen = norm(Ctt - A0*x)/norm(Ctt);
fprintf('AMEn\t\t %.4e\t\t  %.4e\n', res_amen,t_amen)

%--------------------------------------------------------------
% Printing results
printresults = 0; % 1 = print results; 0 = otherwise.
if printresults
    fprintf('\n\n----------------------------------------------------------------\n')
    fprintf('----------------------------------------------------------------\n')
    fprintf('Tk-SS-CG\t  %.4e   \t  %.2f (%d) \n',  res_norm_ss_cg, t_ss_cg, it_ss_cg)
    fprintf('Tk-SS-SD\t  %.4e   \t  %.2f (%d) \n',  res_norm_ss_sd, t_ss_sd, it_ss_sd)
    disp('----------------------------------------------------------------')
    fprintf('P-innerouter Tk-SS-CG\t  %.4e   \t  %.2f (%d) \n',  ...
        res_norm_Pio_ss_cg, t_Pio_ss_cg, it_Pio_ss_cg)
    fprintf('P-innerouter Tk-SS-SD\t  %.4e   \t  %.2f (%d) \n',  ...
        res_norm_Pio_ss_sd, t_Pio_ss_sd, it_Pio_ss_sd)
    disp('----------------------------------------------------------------')
    fprintf('Time to form P-FTT \t\t%.4f\n', t_PFTT)
    fprintf('P-FFT Tk-SS-CG\t  %.4e   \t  %.2f (%d) \n',  ...
        res_norm_PFTT_ss_cg, t_PFTT_ss_cg+t_PFTT, it_PFTT_ss_cg)
    fprintf('P-FTT Tk-SS-SD\t  %.4e   \t  %.2f (%d) \n',  ...
        res_norm_PFTT_ss_sd, t_PFTT_ss_sd+t_PFTT, it_PFTT_ss_sd)
    disp('----------------------------------------------------------------')
    fprintf('Time to form P-EIGs \t\t%.4f\n', t_Peig)
    fprintf('P-EIGs Tk-SS-CG\t  %.4e   \t  %.2f (%d) \n',  ...
        res_norm_Peig_ss_cg, t_Peig_ss_cg+t_Peig, it_Peig_ss_cg)
    fprintf('P-EIGs Tk-SS-SD\t  %.4e   \t  %.2f (%d) \n',  ...
        res_norm_Peig_ss_sd, t_Peig_ss_sd+t_Peig, it_Peig_ss_sd)
    disp('----------------------------------------------------------------')
    fprintf('AMEn\t\t   %.4e  \t %.2f\n', res_amen, t_amen)
end

%--------------------------------------------------------------
% Plot results
plotresults = false; % true = plot results; false = otherwise.
if plotresults
    close all
    % Create figure
    CH_ss_sd = info_tucker_ss_sd{3, 2};
    x_ss_sd = 0:it_ss_sd;
    plot(x_ss_sd, CH_ss_sd(1+x_ss_sd), ...
        'LineStyle', '-', ...         % Solid line
        "LineWidth",1.5,...
        'Marker', 'square', ...            % Circle marker
        'Color', "#0072BD", ...             % Blue line color
        'MarkerFaceColor', "#0072BD", ...   % Filled marker face color (same as line)
        'MarkerSize', 6, ...          % Marker size for better visibility
        'MarkerIndices', 1:2:it_ss_sd,...
        'DisplayName','Tk--SS--SD');

    hold on

    CH_ss_cg = info_tucker_ss_cg{3, 2};
    x_ss_cg = 0:it_ss_cg;
    plot(x_ss_cg, CH_ss_cg(1+x_ss_cg), ...
        'LineStyle', '-', ...         % Solid line
        "LineWidth",1.5,...
        'Marker', 'diamond', ...            % Circle marker
        'Color', "#7E2F8E", ...             % Blue line color
        'MarkerFaceColor', "#7E2F8E", ...   % Filled marker face color (same as line)
        'MarkerSize', 6, ...          % Marker size for better visibility
        'MarkerIndices', 1:2:it_ss_cg,...
        'DisplayName','Tk--SS--CG');

    hold on
    CH_Peig_ss_sd = info_Peig_tucker_ss_sd{3, 2};
    x_Peig_ss_sd = 0:it_Peig_ss_sd;
    plot(x_Peig_ss_sd, CH_Peig_ss_sd(1+x_Peig_ss_sd), ...
        'LineStyle', '-', ...         % Solid line
        "LineWidth",1.5,...
        'Marker', '^', ...            % Circle marker
        'Color', "#D95319", ...             % Blue line color
        'MarkerFaceColor', "#D95319", ...   % Filled marker face color (same as line)
        'MarkerSize', 6, ...          % Marker size for better visibility
        'MarkerIndices', 1:2:it_Peig_ss_sd,...
        'DisplayName','P--Eig Tk--SS--SD');

    hold on
    CH_PFTT_ss_sd = info_PFTT_tucker_ss_sd{3, 2};
    x_PFTT_ss_sd = 0:it_PFTT_ss_sd;
    plot(x_PFTT_ss_sd, CH_PFTT_ss_sd(1+x_PFTT_ss_sd), ...
        'LineStyle', '-', ...         % Solid line
        "LineWidth",1.5,...
        'Marker', 'o', ...            % Circle marker
        'Color', "#77AC30", ...             % Blue line color
        'MarkerFaceColor', "#77AC30", ...   % Filled marker face color (same as line)
        'MarkerSize', 6, ...          % Marker size for better visibility
        'MarkerIndices', 1:2:it_PFTT_ss_sd, ...%)%, ...
        'DisplayName','P--FFT Tk--SS--SD')

    hold on
    CH_Pio_ss_sd = info_Pio_tucker_ss_sd{3, 2};
    x_Pio_ss_sd = 0:it_Pio_ss_sd;
    plot(x_Pio_ss_sd, CH_Pio_ss_sd(1+x_Pio_ss_sd), ...
        'LineStyle', '-', ...         % Solid line
        "LineWidth",1.5,...
        'Marker', 'pentagram', ...            % Circle marker
        'Color', "#EDB120", ...             % Blue line color
        'MarkerFaceColor', "#EDB120", ...   % Filled marker face color (same as line)
        'MarkerSize', 5.5, ...          % Marker size for better visibility
        'MarkerIndices', 1:2:it_Pio_ss_sd,...
        'DisplayName','P--InnOut Tk--SS--SD')

    legend;
    yscale('log')
    % 2. Turn on the grid
    grid on;       % Shows the major lines (10^1, 10^2)
    grid minor;    % Shows the lines in between (20, 30, 40...)
    xlabel('number of iterations'); % Note the $ $ for math mode
    ylabel('relative residual');
    ylim([4e-5 1])
    if m == 10, xlim([0 80]), elseif m == 15,  xlim([0 60]), end
    set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
    set(groot, 'defaultLegendInterpreter','latex');
    ax = gca;
    %ax.GridAlpha = 0.35;      % Transparency of major lines
    ax.MinorGridAlpha = 0.3; % Transparency of minor lines
    %ax.MinorGridLineStyle = '--'; % Transparency of minor lines
    ax.TickLabelInterpreter = 'latex'; % Changes the numbers on axes (0, 0.5, 1...) to LaTeX font
    ax.FontName = 'Latin Modern Math'; % Or 'Times New Roman' depending on your OS
    ax.FontSize = 25; 
    ax.XMinorTick = 'off';
    ax.YMinorTick = 'off';
    ax.Box ="off";
    xline(ax,ax.XLim(2), 'HandleVisibility','off')
    yline(ax,ax.YLim(2), 'HandleVisibility','off')

    set([ax.XLabel, ax.YLabel, ax.Title], 'Interpreter', 'latex');
    set(legend, 'Box', 'off', 'Interpreter', 'latex', 'FontSize', 28, 'Location', 'none', 'Position', [0.53, 0.74, 0.2, 0.001]);
    if iproblem == 5 && n == 1000
        figname = strcat('CHip5_n', num2str(n), '_r', num2str(m), '.eps');  
        exportgraphics(figure(1), figname, 'ContentType', 'vector');
    end
end
