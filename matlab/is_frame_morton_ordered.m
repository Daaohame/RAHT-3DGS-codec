%CONFIDENTIAL (C) Mitsubishi Electric Research Labs (MERL) 2017 Eduardo
%Pavez 08/17/2017
function [ error,out,index ] = is_frame_morton_ordered( Vin,J )
%Checks if the point cloud vertices are sorted by morton code

%get morton code
N=size(Vin,1);
V=floor(double(Vin));
 M=zeros(N,1);
% 
 tt=[1;2;4];
 for i=1:J
     M=M+fliplr(bitget(V,i,'uint64'))*tt;
     
     tt=tt*8;
     
 end
[~,index]=sort(M);

error=norm(V-V(index,:));
 
 out=Vin(index,:);
end

