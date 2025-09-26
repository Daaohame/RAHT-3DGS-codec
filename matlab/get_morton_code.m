% CONFIDENTIAL
% (C) Mitsubishi Electric Research Labs (MERL) 2017
% Eduardo Pavez 08/17/2017

function [ M ] = get_morton_code( V, J )
% V is Nx3 pointcloud with integer coordinates,  J is octree depth
% get morton code

  N = size(V,1);
  M = zeros(N,1);
  tt = [1;2;4];

  for i=1:J
    M = M + fliplr(bitget(V,i,'uint64')) * tt;
    tt = tt*8;
  end

end
