% RAHT Debug Script - Simple 8-point cube test
% For debugging RAHT transform with minimal test case
clear;
addpath(genpath('RA-GFT'));
addpath(genpath('RAHT'));

fprintf('=== RAHT DEBUG MODE ===\n');

% Generate sample colored point cloud
N = 10000;
V = rand(N,3) * 10;                  % xyz in [0,10]
C = randi([0,255], N, 3);            % RGB attributes
PC = [V, C];

% Set voxelization parameters
param.vmin = [0 0 0];                % lower corner of bounding box
param.width = 10;                    % cube side length
param.J = 4;                         % octree depth -> 16^3 voxels
param.writeFileOut = false;          % disable file output
param.filename = 'example';          % base name if file writing enabled

% Voxelize point cloud
[PCvox, PCsorted, voxel_indices, DeltaPC] = voxelizePC(PC, param);

% Extract voxelized and sorted coordinates and attributes
voxel_size = param.width/(2^param.J);
V0s  = PCsorted(:,1:3) - param.vmin;    % sorted coordinates
V0i  = floor(V0s/voxel_size);           % sorted voxel indices
V = V0i(voxel_indices,:);               % Morton-ordered voxel coords
Cvox = PCvox(:,4:end);                  % already Morton-ordered colors
C = RGBtoYUV(Cvox); % Convert to YUV color space

% [error, V_morton, index] = is_frame_morton_ordered(V, param.J);
% disp(error);

J = param.J;

fprintf('Input Configuration:\n');
fprintf('  Number of points: %d\n', size(V,1));
fprintf('  Octree depth J: %d\n', J);
fprintf('  Points (V):\n');
disp(size(V));
fprintf('  Colors (C):\n');
disp(size(C));
fprintf('\n');

% Minimum corner and width
minV = [0, 0, 0];
width = 2^J;

fprintf('Processing parameters:\n');
fprintf('  minV: [%.1f, %.1f, %.1f]\n', minV);
fprintf('  width: %.1f\n', width);
fprintf('  Voxel size Q: %.4f\n', width/2^J);
fprintf('\n');

% Compute RAHT parameters
fprintf('Computing RAHT parameters...\n');
[ListC, FlagsC, weightsC] = RAHT_param(V, minV, width, J);

fprintf('  ListC size: %d\n', length(ListC));
fprintf('  FlagsC size: %d\n', length(FlagsC));
fprintf('  weightsC size: %d\n', length(weightsC));
fprintf('\n');

% Apply RAHT transform
fprintf('Applying RAHT transform...\n');
[Coeff, w] = RAHT(C, ListC, FlagsC, weightsC);

fprintf('  Coefficient matrix size: %d × %d\n', size(Coeff,1), size(Coeff,2));
fprintf('  L2-norm of input C: %.6f\n', norm(C));
fprintf('  L2-norm of Coeff: %.6f\n', norm(Coeff));
fprintf('\n');

% Apply inverse RAHT
fprintf('Applying inverse RAHT...\n');
C_recon = iRAHT(Coeff, ListC, FlagsC, weightsC);

% Verify reconstruction
tolerance = 1e-10;
if ismembertol(C, C_recon, tolerance)
    fprintf('✓ Reconstruction check: PASSED (within tolerance %.0e)\n', tolerance);
else
    fprintf('✗ Reconstruction check: FAILED (difference exceeds tolerance)\n');
end

% Compute reconstruction error
recon_error = norm(C - C_recon, 'fro');
max_error = max(abs(C(:) - C_recon(:)));
fprintf('  Frobenius norm error: %.10e\n', recon_error);
fprintf('  Max absolute error: %.10e\n', max_error);
fprintf('\n');

% Save results for further analysis
save('../results/debug_coeff.mat', 'Coeff');
save('../results/debug_params.mat', 'ListC', 'FlagsC', 'weightsC');
save('../results/debug_input.mat', 'V', 'C', 'J');
save('../results/debug_recon.mat', 'C_recon', 'recon_error', 'max_error');

fprintf('\n=== Debug data saved to ../results/ ===\n');
fprintf('Files: debug_coeff.mat, debug_params.mat, debug_input.mat, debug_recon.mat\n');