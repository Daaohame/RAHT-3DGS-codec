function [T]=iRAHT(Coeff,List,Flags,weights)

T=Coeff;

%%%%%inverse raht
Nlevels=length(Flags);

for j=Nlevels:-1:1
        
        %pick left nodes that have a sibling
       left_sibling_index=Flags{j};
       right_sibling_index=[0;Flags{j}(1:end-1)];
       %
        i0=List{j}(left_sibling_index==1);
        i1=List{j}(right_sibling_index==1);
        if(~isempty(i0) && ~isempty(i1))
        %pick coefficients corresponding to sets i0, and i1
        x0=T(i0,:);
        x1=T(i1,:);
        signal_dimension=size(T,2);
        %pick transform weights
        w0=weights{j}(left_sibling_index==1);
        w1=weights{j}(right_sibling_index==1);
        %compute 2x2 raht matrix 
        a=sqrt(w0./(w0+w1));
        b=sqrt(w1./(w0+w1));
       % w(i0)=w(i0)+w(i1); %unnecessary for inverse
       % w(i1)=w(i0);
        T(i0,:)=repmat(a,1,signal_dimension).*x0-repmat(b,1,signal_dimension).*x1;
        T(i1,:)=repmat(b,1,signal_dimension).*x0+repmat(a,1,signal_dimension).*x1;
        end
    
end


end