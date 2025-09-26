function [V, C] = read_ply_file(filename)
%READ_PLY_FILE Read vertices (and color/intensity) from a .ply file.
%   Uses Computer Vision Toolbox's pcread; supports ASCII & binary PLY.
%
%   Args:
%       filename (char/string): path to .ply
%
%   Returns:
%       V (double, N×3): xyz coordinates
%       C (double, N×K): attributes; raw RGB values (0–255) if present (K=3),
%                        otherwise intensity as N×1. Empty [] if none.
%
%   Notes:
%       - Requires Computer Vision Toolbox (pcread).
%       - Only RGB color (properties red/green/blue) and Intensity are
%         available via pointCloud; arbitrary extra vertex fields are not
%         returned by this helper.

    % ---- Validate input ----
    if nargin ~= 1
        error('read_ply_file:BadInputs', 'Usage: [V,C] = read_ply_file(filename)');
    end
    if ~(ischar(filename) || (isstring(filename) && isscalar(filename)))
        error('read_ply_file:BadInputs', 'filename must be a char vector or scalar string.');
    end
    filename = char(filename);

    % ---- Existence checks ----
    if ~exist(filename, 'file')
        error('read_ply_file:NotFound', 'File not found: %s', filename);
    end
    if exist('pcread', 'file') ~= 2
        error('read_ply_file:CVTRequired', ...
              'Computer Vision Toolbox is required (pcread not found).');
    end

    % ---- Read with pcread ----
    try
        pc = pcread(filename);
    catch ME
        % Re-throw with clearer context
        ME2 = MException('read_ply_file:ReadFailed', ...
             'Failed to read PLY "%s" with pcread: %s', filename, ME.message);
        ME2 = addCause(ME2, ME);
        throw(ME2);
    end

    % ---- Extract & normalize outputs ----
    V = double(pc.Location);  % always double

    if ~isempty(pc.Color)
        % Preserve raw RGB, just convert to double (0–255)
        C = double(pc.Color);
    elseif isprop(pc, 'Intensity') && ~isempty(pc.Intensity)
        C = double(pc.Intensity);
        % Ensure N×1 shape
        if isvector(C), C = C(:); end
        if size(C,1) ~= size(V,1)
            error('read_ply_file:SizeMismatch', ...
                  'Intensity length (%d) does not match vertex count (%d).', size(C,1), size(V,1));
        end
    else
        C = [];
    end
end