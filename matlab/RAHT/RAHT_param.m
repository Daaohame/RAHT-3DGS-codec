function [List,Flags,weights]=RAHT_param(V,minV,width,depth)
%computes index list, and flags for RAHT and iRAHT transforms
%V: quantized and voxelized(morton order)

%compute morton code

Q=width/2^depth;
sizeV=size(V,1);
Vint= floor((V-repmat(minV,sizeV,1))/Q);


MC=zeros(sizeV,1);
tri=[1;2;4];


for i=1:depth

    MC=MC+fliplr(bitget(Vint,i,'uint64'))*tri;
    
    tri=8*tri;

end


 %%%%%%now create list and flag arrays
    Nbits=3*depth;
    List=cell(Nbits,1);
    Flags=cell(Nbits,1);
    %weight array
    weights=cell(Nbits,1);
    %initialize list
    List{1}=(1:1:sizeV)';
    %64=max number of bits of an integer in matlab (uint64)
    for j=1:64
        
        %compute weights
        weights{j}=[List{j}(2:end);sizeV+1]-List{j};
        
        
        Mj=uint64(MC(List{j}));
        
        diff=bitxor(Mj(1:end-1),Mj(2:end),'uint64');% put a 1 in i-th position if i-th bits are different
        
        %check if Nbits-j most significant bits are equal, have same prefix
        %at level j
        masked=bitand(diff,uint64(2^Nbits-2^j));%2^Nbits-1 -(2^j-1)
        
        Flags{j}=[masked==0;0];
        
        tmpList=List{j}(~[0;Flags{j}(1:end-1)]);
        if(size(tmpList,1)==1)
            break
        end
        List{j+1}=tmpList;
        
    end






end