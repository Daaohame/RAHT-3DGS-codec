function [T,w]=RAHT(C,List,Flags,weights)

T=C; % [N, number of attribute]
w=ones(size(C,1),1);
Nlevels=length(Flags); % of the octree

for j=1:Nlevels % bottom up
    
    % pick left nodes that have a sibling
    left_sibling_index=Flags{j};    % a binary list indicating whether a node is a left sibling to another node
    right_sibling_index=[0;Flags{j}(1:end-1)];

    % get actual indices of the nodes
    i0=List{j}(left_sibling_index==1);
    i1=List{j}(right_sibling_index==1);
    if(~isempty(i0) && ~isempty(i1))
        %pick coefficients corresponding to sets i0, and i1
        x0=T(i0,:);
        x1=T(i1,:);
        signal_dimension=size(T,2); % number of attribute
        %pick transform weights
        w0=weights{j}(left_sibling_index==1);
        w1=weights{j}(right_sibling_index==1);
        %compute 2x2 raht matrix
        a=sqrt(w0./(w0+w1));
        b=sqrt(w1./(w0+w1));
        w(i0)=w(i0)+w(i1);
        w(i1)=w(i0);
        T(i0,:)=repmat(a,1,signal_dimension).*x0+repmat(b,1,signal_dimension).*x1;
        T(i1,:)=-repmat(b,1,signal_dimension).*x0+repmat(a,1,signal_dimension).*x1;
    end
end

end