%run RAHT + RLGR coder for attributes
clear;
addpath(genpath('RA-GFT'));
addpath(genpath('RAHT'));

function J_adj = raht_adjust_depth_increase_only(V, minV, width, J_req)
% Choose a depth that fits data; only increases J if needed.
% V: N×3 double, minV: 1×3, width: scalar (cube side), J_req: requested depth

    % voxel size for requested depth
    Q = width / 2^J_req;

    % quantize with a tiny tolerance to avoid boundary spillover
    Vint = floor( (V - minV) / Q - 1e-12 );

    % maximum index across all axes
    maxIndex = max(Vint(:));              % could be negative if minV>min(V), but that’s fine
    needed   = ceil(log2(double(maxIndex + 1)));  % bits needed per axis

    % if data has no extent (needed<=0), depth=0 is enough
    needed = max(0, needed);

    % only increase if needed
    J_adj = max(J_req, needed);

    % cap for uint64 Morton codes (3*J <= 63)
    J_adj = min(J_adj, 21);
end

% List of PLY file paths
ply_paths = {'/ssd1/haodongw/workspace/3dstream/3DGS_Compression_Adaptive_Voxelization/attributes_compressed/train_depth_15_thr_30_3DGS_adapt_lossless/train_dc.ply'};
J = 15;

T = length(ply_paths);
Nvox = zeros(T,1);
time = Nvox;

for frame =1:T
    tic;

    [ V, C ] = read_ply_file(ply_paths{frame});
    N = size(V,1);
    Nvox(frame) = N;

    J_adj = raht_adjust_depth_increase_only(V, [0 0 0], 2^J, J);

    % ListC: indices of colors
    % FlagsC: indicator of whether a node is a left sibling to another node
    % weightsC: weights of colors
    [ListC,FlagsC,weightsC]=RAHT_param(V,[0 0 0], 2^J_adj, J_adj);
    
    [Coeff,w]=RAHT(C,ListC,FlagsC,weightsC);
    C_recon = iRAHT(Coeff, ListC, FlagsC, weightsC);

    % Save for debug
    save(sprintf('../results/frame%d_coeff_matlab.mat', frame), 'Coeff');
    save(sprintf('../results/frame%d_params_matlab.mat', frame), 'ListC', 'FlagsC', 'weightsC');

    if ismembertol(C, C_recon, 1e-8)
        fprintf('Reconstruction check: PASSED (within tolerance)\n');
    else
        fprintf('Reconstruction check: FAILED (difference exceeds tolerance)\n');
    end

    fprintf('L2-norm of C: %.4f\n', norm(C));
    fprintf('L2-norm of Coeff: %.4f\n', norm(Coeff));
    time(frame)=toc;
    disp([time(frame) frame T])
end


