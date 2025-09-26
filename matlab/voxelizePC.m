%CONFIDENTIAL (C) Mitsubishi Electric Research Labs (MERL) 2017 Eduardo
%Pavez 08/17/2017
function [ PCvox, PCsorted,voxel_indices, DeltaPC ] = voxelizePC( PC, param )
%   PC(:,1:3)=V: vertices N x 3 matrix, each row is xyz coords of a point
%   PC(:,4:end)=C: colors (or attributes) N x d matrix
%   vmin: xyz coordinates of smaller point, this can be provided, if many
%   point clouds will be voxelized wrt the same bounding box, or it can be
%   pointcloud dependent
%   width: width of assumed cubic bounding box, i.e. points belong to width
%   x width x width cube
%   J: max depth of octree decomposition, voxels are of dimension 
%   width/2^J x  width/2^J x  width/2^J
vmin = param.vmin;
width = param.width;
J = param.J;
writeFileOut = param.writeFileOut;
filename = param.filename;
hasAttribute = size(PC,2)>3 ;
V = PC(:,1:3);
if(hasAttribute)
  C = PC(:,4:end);
end

if(isempty(vmin))
  vmin = min(V);
end
N = size(PC,1);
V0 = V - repmat(vmin,N,1);

if(isempty(width))
  width = max(V0(:));
end


%sort original PC in ascending morton code
voxel_size = width/(2^J);
V0_integer = floor(V0/voxel_size);
M  = get_morton_code( V0_integer, J );

[M_sort, idx] = sort(M,'ascend');% M(idx)=M_sort;
V0 = V0(idx,:);
PCsorted = V(idx,:);
if(~isempty(C))
  C0 = C(idx,:);
  PCsorted = [PCsorted,C0];
end
V0voxelized = voxel_size*(floor(V0/voxel_size)); %the coordinates have been quantized to voxel mid point
DeltaV = V0 - V0voxelized;                           %this is a quantization error
%
voxel_boundary = M_sort(2:end) - M_sort(1:end-1); % there is a nonzero when two consecutive points belong to different voxel
voxel_indices = find([1;voxel_boundary]);         %indices of voxel boundary, i.e. points in voxel_indices(i):(voxel_indices(i+1)-1) belong to same voxel
%voxel_Npoints = [voxel_indices(2:end);N] - voxel_indices; % number of points that fall in the same voxel,  voxel_Npoints(i) = length(voxel_indices(i):(voxel_indices(i+1)-1))

%now create voxelized point cloud

Nvox = size(voxel_indices,1); %number of voxels
if(~isempty(C))
  C0voxelized = zeros(size(C0));
  for i=1:Nvox
    start_ind = voxel_indices(i);
    if(i==Nvox)
      end_ind = N;
    else
      end_ind = voxel_indices(i+1)-1;
    end
    
    Ni = length(start_ind:end_ind);
    cmean = mean(C0(start_ind:end_ind,:),1);
    C0voxelized(start_ind:end_ind,:) = repmat(cmean,Ni,1); %replace each attribute, by the average of attributes in the same voxel
  end
end

DeltaC = C0-C0voxelized;

Vvox = V0_integer(voxel_indices,:);
Cvox = C0voxelized(voxel_indices,:);

PCvox = [Vvox, Cvox];
DeltaPC = [DeltaV, DeltaC];
if(writeFileOut)
  
  pc = pointCloud(single(Vvox),'color',uint8(Cvox));%there is some loss by converting to uint8
  filename_pcvox = sprintf('%s_vox.ply',filename);
  pcwrite(pc,filename_pcvox);
  
  filename_data = sprintf('%s_data.txt',filename);
  fileID = fopen(filename_data,'w');
  fprintf(fileID,'%f %f %f %f %d %d %d %d\n',vmin(1),vmin(2), vmin(3), width, J, Nvox, N, hasAttribute);%vmin, width, J
  fprintf(fileID,'%d\n',voxel_indices);
  if(hasAttribute)
    fprintf(fileID,'%f %f %f %f %f %f\n',DeltaPC);
  else
    fprintf(fileID,'%f %f %f\n',DeltaPC);
  end
  fclose(fileID);
end
% paramOut = param;
% paramOut.Nvox = Nvox;
% paramOut.Npoints = N;

end

