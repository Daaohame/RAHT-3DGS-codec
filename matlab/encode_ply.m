%run RAHT + RLGR coder for attributes
clear;
addpath(genpath('RA-GFT'));
addpath(genpath('RAHT'));

% List of PLY file paths
ply_paths = {'/ssd1/haodongw/workspace/3dstream/3DGS_Compression_Adaptive_Voxelization/attributes_compressed/train_depth_15_thr_30_3DGS_adapt_lossless/train_dc.ply'};
J = 15;

T = length(ply_paths);
Nvox = zeros(T,1);
time = Nvox;

for frame =1:T
    tic;

    [ V,Crgb ] = read_ply_file(ply_paths{frame});
    N = size(V,1);
    Nvox(frame) = N;
    C = Crgb; % shape [N x k]

    disp(size(V));
    disp(size(C));
    
    % ListC: indices of colors
    % FlagsC: indicator of whether a node is a left sibling to another node
    % weightsC: weights of colors
    [ListC,FlagsC,weightsC]=RAHT_param(V,[0 0 0], 2^J, J);
    
    [Coeff,w]=RAHT(C,ListC,FlagsC,weightsC);
    [~,IX_ref] = sort(w,'descend');
    
    Y = Coeff(:,1);

    fprintf('Energy of C: %.4f\n', norm(C));
    fprintf('Energy of Coeff: %.4f\n', norm(Coeff));
    time(frame)=toc;
    disp([time(frame) frame T])
end


