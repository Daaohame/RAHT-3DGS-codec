function [List,Flags,weights]=RAHT_param(V,minV,width,depth)
% computes index list, and flags for RAHT and iRAHT transforms
% V: quantized and voxelized(morton order) is derived here from V (double)

    Q     = width/2^depth;
    sizeV = size(V,1);

    % Quantize to voxel indices (expect in-range)
    Vint = floor((V - repmat(minV,sizeV,1)) / Q);
    if any(Vint(:) < 0) || any(Vint(:) > 2^depth - 1)
        error('RAHT_param:OutOfBounds', ...
            'Quantized indices must be within [0, 2^depth-1] per axis. Check minV/width/depth.');
    end
    Vint = uint64(Vint);

    %%% compute morton code (uint64, no integer matrix multiply)
    MC = zeros(sizeV,1,'uint64');
    for i = 1:depth
        b     = bitget(Vint, i);  % N×3 (x,y,z), 0/1 as double
        digit = uint64(b(:,3)) + bitshift(uint64(b(:,2)),1) + bitshift(uint64(b(:,1)),2);
        MC    = bitor(MC, bitshift(digit, 3*(i-1)));
    end

    %%% create list and flag arrays
    Nbits   = 3*depth;
    List    = cell(Nbits,1);
    Flags   = cell(Nbits,1);
    weights = cell(Nbits,1);

    List{1} = (1:sizeV)';

    % 64 = max bits of uint64; break when list collapses
    for j = 1:64
        % run-length weights
        weights{j} = [List{j}(2:end); sizeV+1] - List{j};

        Mj = MC(List{j});

        if numel(Mj) == 1
            Flags{j} = false;
            break
        end

        % bit differences between adjacent Morton codes
        diff   = bitxor(Mj(1:end-1), Mj(2:end));     % uint64
        mask   = bitshift(uint64(1), Nbits) - bitshift(uint64(1), j);  % 2^Nbits - 2^j, as uint64
        masked = bitand(diff, mask);

        Flags{j} = [masked == 0; false];

        tmpList = List{j}(~[false; Flags{j}(1:end-1)]);
        if numel(tmpList) == 1
            break
        end
        List{j+1} = tmpList;

        if j >= Nbits
            break
        end
    end
end