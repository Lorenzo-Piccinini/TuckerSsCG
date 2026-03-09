function [X, info] = tucker_sscg_pc(A, C, M, options)
%--------------------------------------------------------------------------
% Tucker subspace Conjugate Gradient method with preconditioning for 
% for solving tensor equations of the form
%
%   \sym_{i=1}^{l} X x_1 A{i,1} x_2 A{i,2} x_3 ... x_d A{i,d} = C
%
% INPUT:
%   A       - [cell array] of size (l, d) containing the factors of the
%               operator
%   C       - [structure] Tucker tensor containing the right-hand side where
%               C.core is a [numeric array] for the core tensor of size 
%               r_1 x r_2 x ... x r_d
%               C.factors is a [cell array] of the Tucker factor matrices 
%               of size (m_k x r_k)
%   M       - [structure] preconditioner; if empty, no preconditioner is 
%               applied, else:
%                M.fftprec: [boolean], if true, inverse of Laplacian 
%                applied via FFT
%                   M.filter: [cell array], 2q+1 vecotrs of exponential of
%                   the scaled eigenvalues of 1-D Laplacian
%                   M.c: [vector] exponential sum coefficients
%               M.fftprec: [boolean], if false, inverse of Laplacian 
%                applied via eigenvalue decomposition
%                   M.E: [cell array], of size (2q+1, d) with matrices of
%                   exponential of the scaled eigenvalues of 1-D Laplacian
%                   M.c: [vector] exponential sum coefficients
%   options - [structure] with fields
%               tol: [float] convergence tolerance for the relative residual
%               maxit: [integer] maximum number of iterations
%               tol_tr: [float] truncation tolerance for the tucker truncation
%               maxrank: [integer] maximum rank for the Tucker truncation
%               innerouter: [boolean] for inner-outer preconditioning
%               innsolver: [integer] for inner solver (0: direct, 1: pc
%                   2: subspaceCG)
%               single_flag: [boolean] for single-precision for preconditioning
%                    and alpha computation
%               X0: [structure] initial guess for the solution (tucker tensor)
%                   X0.core is a [numeric array] for the core tensor of size 
%                   r_1 x r_2 x ... x r_d
%                   X0.factors is a [cell array] of the Tucker factor 
%                   matrices of size (m_k x r_k)
%
% OUTPUT:
%   X       - [structure] Tucker tensor containing the right-hand side where
%               X.core is a [numeric array] for the core tensor of size 
%               r_1 x r_2 x ... x r_d where r_j <= maxrank
%               X.factors is a [cell array] of the Tucker factor matrices 
%               of size (m_k x r_k)            
%  info     - [structure] containing additional information about the 
%               iteration, i.e., last computed coefficients alpha, beta 
%               Residual (Res), direction matricies (P), iterations (it)
%               residual norm convergence history (resvec)
%
% If you use this code, please cite the following paper:
%
% M. Iannacito, L. Piccinini, V. Simoncini, "Subspace gradient descent method for linear tensor equations", arXiv preprint arXiv:2602.21974, 2026.
%
% @article{iannacito2026subspace,
%   title={Subspace gradient descent method for linear tensor equations},
%   author={Iannacito, Martina and Piccinini, Lorenzo and Simoncini, Valeria},
%   journal={arXiv preprint arXiv:2602.21974},
%   year={2026}
% }
% 
% https://arxiv.org/abs/2602.21974
%--------------------------------------------------------------------------

tol = options.tol;
maxit = options.maxit;
tol_tr = options.tol_tr;
maxrank = options.maxrank;
innerouter = options.innerouter;
innsolver = options.innsolver;
single_flag = options.single_flag;
X0 = options.X0;


if innsolver == 0, disp('direct'), elseif innsolver == 1, disp('pcg'),
elseif innsolver == 2, disp('subspaceCG'), else disp('error')
end

if isstruct(M) 
    disp('preconditioner as exp sum (type_prec = 1)')
    type_prec = 1; %
elseif innerouter %isa(M, "function_handle")
    disp('preconditioner as handle funcion (type_prec = 2)')
    type_prec = 2;
elseif isempty(M)
    disp('NO preconditioner (type_prec = 0)')
    type_prec = 0;
else
    disp('error')
    return
end
%----------------------------------------
% Storing arrays and initial values
it = 0;                                % Intial iteration
nterms = size(A,1);                    % number of terms
d = size(A, 2);                        % Order of the problem
%------------------------------

Cfact = C.factors;
Ccore = C.core;

clearvars C
% Initializing the residual and P
if X0 == 0 % Assuming X0 = 0:
    clear X0;
    Res_core = Ccore; % Residual as TT-vec
    Res_fact = Cfact; % Residual as TT-vec
else
    Xcore = X0.core;
    Xfact = X0.factors;
    clear X0;
    return
end

normRes0 = norm(Res_core, 'fro');
rel_res = 1;
resvec = zeros(maxit,1);
resvec(1) = 1;

if type_prec == 0 % No Preconditioning
    P = Cfact;
elseif type_prec == 1 % Preconditioning with Exponential sum
    if M.fftprec % Prec built using FFT
        disp('Prec FFT')
        if single_flag, [~, P] = tucker_S_Mx_trunc_FFTexp(M, Res_core, Res_fact, maxrank, tol_tr); % SINGLE
        else, [~, P] = tucker_Mx_trunc_FFTexp(M, Res_core, Res_fact, maxrank, tol_tr); end % DOUBLE
    else % Prec built using Eigs
        if single_flag, [~, P] = tucker_S_Mx_trunc_exp(M, Res_core, Res_fact, maxrank, tol_tr); % SINGLE
        else, [~, P] = tucker_Mx_trunc_exp(M, Res_core, Res_fact, maxrank, tol_tr); end % DOUBLE
    end
elseif  type_prec == 2 % Inner Outer Preconditioning
    ccc.core=Res_core;ccc.factors=Res_fact;
    options1.maxit = 3; options1.tol = 1e-1; options1.tol_tr = 1e-16;
    options1.maxrank = 10; options1.maxrankP = (maxrank)*ones(d,1);
    options1.innsolver = 2; options1.innerouter = 0; options1.X0 = 0;
    options1.single_flag = true;
    [wrk, ~] = tucker_sssd_pc(A, ccc, [], options1);
    P = cell(1,d);
    for k=1:d, P{k}=wrk.factors{k};end
end

sz_p = zeros(1,d);
for k = 1:d, sz_p(k) = size(P{k}, 2); end

while (rel_res > tol)  && (it <= maxit)
    PtA = cell(nterms,d);
    PtAP = cell(nterms,d);
    PtRes_fact = cell(d,1);
    for k = 1:d
        for l = 1:nterms
            PtA{l,k} = P{k}'*A{l,k};
            PtAP{l,k} = PtA{l,k}*P{k};
        end
        PtRes_fact{k} = P{k}'*Res_fact{k};
    end

    if innsolver == 0 || min(sz_p)==1  % BACKSLASH
        alpha_vec = tucker_OP_full(PtAP)\(reshape(tucker_times_matrix(Res_core,PtRes_fact), [], 1));
        alpha = reshape(alpha_vec, sz_p);
    elseif innsolver == 1 % PCG
        if it == 0 || numel((alpha)) ~=  maxrank*d, alpha = 0; end
        prodPtAP = @(x) tucker_Ax_trunc(PtAP, x, maxrank, 1e-12);
        if single_flag %SINGLE
            rhs = single(reshape(tucker_times_matrix(Res_core, PtRes_fact), [], 1));
            [alpha_vec, ~, ~, ~, ~] = pcg(prodPtAP, rhs, 1e-2, 100);    
            alpha = double(reshape(alpha_vec, sz_p));
        else %DOUBLE
            rhs = reshape(tucker_times_matrix(Res_core, PtRes_fact), [], 1);
            [alpha_vec, ~, ~, ~, ~] = pcg(prodPtAP, rhs, 1e-2, 100);    
            alpha = reshape(alpha_vec, sz_p);
        end
    elseif innsolver == 2 % SubspaceCG on matricization
        if single_flag % SubspaceCG in SINGLE precision
            matPtAP = cell(2, nterms);
            RHS_l = single(PtRes_fact{1});
            RHS_c = single(reshape(Res_core, size(Res_core, 1), []));
            RHS_r = kron(single(PtRes_fact{3}), single(PtRes_fact{2}));
            for ll = 4:d
                RHS_r = kron(single(PtRes_fact{ll}), RHS_r);
            end
            for kk = 1:nterms
                matPtAP{1, kk} = single(PtAP{kk, 1});
                matPtAP{2, kk} = kron(single(PtAP{kk, 3}), single(PtAP{kk, 2}));
                for ll = 4:d
                    matPtAP{2, kk} = kron(single(PtAP{kk, ll}), matPtAP{2, kk});
                end
            end
            type_res=2;
            tol_inner=min(1e-5,rel_res);
            [alpha_l, alpha_c, alpha_r, ~] = sscg_mod(matPtAP, {RHS_l, RHS_c, RHS_r}, ...
                tol_inner, 100, tol_tr, (maxrank), d*(maxrank), [], 3, type_res);
            alpha = double(reshape(alpha_l*alpha_c*alpha_r', sz_p));
        else % Subspace CG in DOUBLE precision
            RHS_l = (PtRes_fact{1});
            RHS_c = (reshape(Res_core, size(Res_core, 1), []));
            RHS_r = kron((PtRes_fact{3}), (PtRes_fact{2}));
            for ll = 4:d
                RHS_r = kron((PtRes_fact{ll}), RHS_r);
            end
            matPtAP = cell(2, nterms);
            for kk = 1:nterms
                matPtAP{1, kk} = (PtAP{kk, 1});
                matPtAP{2, kk} = kron((PtAP{kk, 3}), (PtAP{kk, 2}));
                for ll = 4:d
                    matPtAP{2, kk} = kron((PtAP{kk, ll}), matPtAP{2, kk});
                end
            end
            type_res=2;
            tol_inner=min(1e-5,rel_res);
            [alpha_l, alpha_c, alpha_r, ~] = sscg_mod(matPtAP, {RHS_l, RHS_c, RHS_r}, ...
                tol_inner, 100, tol_tr, (maxrank), d*(maxrank), [], 3, type_res);
            alpha = (reshape(alpha_l*alpha_c*alpha_r', sz_p));
        end
    else
        disp('alpha error, exiting...')
        break
    end

    % Updating X
    if it == 0
        Xfact = cell(1,d);
        for k = 1:d
            Xfact{k} = P{k};
        end
        Xcore = alpha;
    else
        upXcore = tucker_blkdiag(Xcore, alpha);
        upXfact = cell(1,d);
        for k = 1:d
            upXfact{k} = [Xfact{k}, P{k}];
            [Ux, Sx, Vx] = svd(upXfact{k}, 'econ');
            sx = diag(Sx)./sum(diag(Sx));
            rx = (sum(sx > tol_tr));
            Xfact{k} = Ux(:, 1:rx);
            upXcore = tucker_times_matrix(upXcore, (Sx(1:rx, 1:rx)*(Vx(:, 1:rx)')), k);
        end
        [Xcore, newXfact] = tucker_trunc_dense((upXcore), maxrank, tol_tr);
        for k=1:d
            Xfact{k} = Xfact{k}*newXfact{k};
            newXfact{k} = [];
        end
    end

    % Updating Residual
    Res_core = Ccore;
    Res_fact = Cfact;
    for l = 1:nterms
       Res_core = tucker_blkdiag((Res_core), -Xcore);
       for k = 1:d  
            Res_fact{k} = [Res_fact{k}, A{l, k}*Xfact{k}];
            [Ures, Sres, Vres] = svd(Res_fact{k}, 'econ');
            sres = diag(Sres)./sum(diag(Sres));
            rres = (sum(sres > tol_tr));
            Res_fact{k} = Ures(:, 1:rres);
            Res_core = tucker_times_matrix(Res_core, (Sres(1:rres, 1:rres)*(Vres(:, 1:rres)')), k);
       end
    end
    [Res_core, newRes_fact] = tucker_trunc_dense((Res_core), maxrank, tol_tr);
    for k=1:d
        Res_fact{k} = Res_fact{k}*newRes_fact{k};
        newRes_fact{k} = [];
    end
    
    % Computing residual's norm
    rel_res = norm(Res_core, 'fro')/normRes0;
    resvec(it+2) = rel_res;
    fprintf('it: %d, rel res it: %.3e\n', it, rel_res)
    % Checking exit tolerance
    if rel_res < tol
        X.core = Xcore;
        X.factors = Xfact;
        it = it+1;
        break
    end
    
    PtARes_fact = cell(1, nterms);
    for l = 1:nterms
        PtARes_fact{l} = cell(1, d);
        for k = 1:d
            PtARes_fact{l}{k} = PtA{l,k}*Res_fact{k};
        end
    end

    % Computing beta
    if innsolver == 0 || min(sz_p)==1 % BACKSLASH
        rhs = reshape(tucker_times_matrix(Res_core, PtARes_fact{1}), [], 1);
        for l = 2:nterms
            rhs = rhs +reshape(tucker_times_matrix(Res_core,PtARes_fact{l}), [], 1);
        end
        beta_vec = tucker_OP_full(PtAP)\-rhs;
        beta = reshape(beta_vec, sz_p);
    elseif innsolver == 1 % PCG
        if it == 0 || numel(beta) ~= maxrank*d, beta = 0; end
        if single_flag  % SINGLE
            rhs = single(reshape(tucker_times_matrix(Res_core, PtARes_fact{1}), [], 1));
            for l = 2:nterms
                rhs = rhs + single(reshape(tucker_times_matrix(Res_core,PtARes_fact{l}), [], 1));
            end
            [beta_vec, ~, ~, ~, ~] = pcg(prodPtAP, -reshape(rhs,[], 1), ...
                1e-2, 100);
            beta = double(reshape(beta_vec, sz_p));
        else % DOUBLE
            rhs = reshape(tucker_times_matrix(Res_core, PtARes_fact{1}), [], 1);
            for l = 2:nterms
                rhs = rhs + reshape(tucker_times_matrix(Res_core,PtARes_fact{l}), [], 1);
            end
            [beta_vec, ~, ~, ~, ~] = pcg(prodPtAP, -reshape(rhs,[], 1), ...
                1e-2, 100);
            beta = reshape(beta_vec, sz_p);
        end
    elseif innsolver == 2 % subspaceCG on maticization of reduced problem along mode1
        if single_flag % SINGLE
            rhs = single(reshape(tucker_times_matrix(Res_core, PtARes_fact{1}), [], 1));
            for l = 2:nterms
                rhs = rhs + single(reshape(tucker_times_matrix(Res_core,PtARes_fact{l}), [], 1));
            end
            [RHScore, RHS_fact] = tucker_trunc_dense(reshape(rhs, sz_p), maxrank, tol_tr);
            RHS_l = (RHS_fact{1});
            RHS_c = reshape(RHScore, size(RHScore, 1), []);
            RHS_r = kron(RHS_fact{3}, RHS_fact{2});
            for kk=4:d
                RHS_r = kron(RHS_fact{kk}, RHS_r);
            end
            [beta_l, beta_c, beta_r, ~] = sscg_mod(matPtAP, {(-1)*RHS_l, RHS_c, RHS_r}, ...
                tol_inner, 100, tol_tr, maxrank, d*maxrank, [], 3, type_res);         
            beta = double(reshape(beta_l*beta_c*beta_r', sz_p));
        else % DOUBLE
            rhs = (reshape(tucker_times_matrix(Res_core, PtARes_fact{1}), [], 1));
            for l = 2:nterms
                rhs = rhs + (reshape(tucker_times_matrix(Res_core,PtARes_fact{l}), [], 1));
            end
            [RHScore, RHS_fact] = tucker_trunc_dense(reshape(rhs, sz_p), maxrank, 1e-12);
            RHS_l = (RHS_fact{1});
            RHS_c = reshape(RHScore, size(RHScore, 1), []);
            RHS_r = kron(RHS_fact{3}, RHS_fact{2});
            for kk=4:d
                RHS_r = kron(RHS_fact{kk}, RHS_r);
            end
            [beta_l, beta_c, beta_r, ~] = sscg_mod(matPtAP, {(-1)*RHS_l, RHS_c, RHS_r}, ...
                tol_inner, 100, tol_tr, maxrank, d*maxrank, [], 3, type_res);
            beta = (reshape(beta_l*beta_c*beta_r', sz_p));
        end
    else
        disp('beta error, exiting...')
        break
    end

    % Update P
    gamma_fact = cell(1,d);
    for k = 1:d
        gamma_fact{k} = [Res_fact{k}, P{k}];
    end
   
    if type_prec == 0 % No Preconditioning
        gamma_core = tucker_blkdiag(Res_core, beta);
        gamma_fact = cell(1,d);
        for k = 1:d
            gamma_fact{k} = [Res_fact{k}, P{k}];
            [Ug, Sg, Vg] = svd(gamma_fact{k}, 'econ');
            sg = diag(Sg)./sum(diag(Sg));
            rg = sum(sg>tol_tr);
            gamma_fact{k} = Ug(:, 1:rg);
            gamma_core = tucker_times_matrix(gamma_core, (Sg(1:rg, 1:rg)*(Vg(:,1:rg)')), k);
        end
        [~, new_gamma_fact] = tucker_trunc_dense((gamma_core), maxrank, tol_tr);
        P = cell(1,d);
        sz_p = zeros(1,d);
        for k=1:d
            P{k} = gamma_fact{k}*new_gamma_fact{k};
            sz_p(k) = size(P{k},2); 
            gamma_fact{k} = [];
            new_gamma_fact{k} = [];
        end
    elseif type_prec == 1 % Preconditioning with exponential sum
        gamma_core = tucker_blkdiag(Res_core, beta);
        if M.fftprec % Prec built with FFT
             if single_flag, [~, P] = tucker_S_Mx_trunc_FFTexp(M, gamma_core, gamma_fact, maxrank, tol_tr); % SINGLE
             else, [~, P] = tucker_Mx_trunc_FFTexp(M, gamma_core, gamma_fact, maxrank, tol_tr); end % DOUBLE
        else % Prec built with Eigs
            if single_flag, [~, P] = tucker_S_Mx_trunc_exp(M, gamma_core, gamma_fact, maxrank, tol_tr); % SINGLE
            else, [~, P] = tucker_Mx_trunc_exp(M, gamma_core, gamma_fact, maxrank, tol_tr); end % DOUBLE
        end
         sz_p = zeros(1,d);
        for k=1:d, sz_p(k) = size(P{k},2); end
    elseif type_prec == 2 % Inner Outer Preconditioning
        ccc.core = tucker_blkdiag(Res_core, beta);
        ccc.factors = gamma_fact;
        [wrk, ~] = tucker_sssd_pc(A, ccc, [], options1);
        for k=1:d
            P{k}=wrk.factors{k};
            sz_p(k) = size(P{k},2); 
        end
    end
    it = it + 1;

end

if it == maxit+ 1
    X.core = Xcore;
    X.factors = Xfact;
end
Res.core = Res_core; Res.factors = Res_fact;
info = {alpha, beta; Res, P; it, resvec};
