%run RAHT + RLGR coder for attributes
clear;
addpath(genpath('RA-GFT'));
addpath(genpath('RAHT'));

% % Voxelized ptcl dataset
% dataset = 'MVUB';
% sequence =  'andrew9';
% sequence =  'david9';
% sequence =  'phil9';
% sequence =  'ricardo9';
% sequence =  'sarah9';

% Voxelized ptcl dataset
dataset='8iVFBv2';
sequence = 'redandblack';
% sequence = 'soldier';
% sequence = 'longdress';
% sequence = 'loot';

T = get_pointCloud_nFrames(dataset, sequence);

colorStep = [1 2 4 8 16 32 64];
colorStep = [1 2 4 6 8 12 16 20 24 32 64];
nSteps = length(colorStep);
bytes = zeros(T,nSteps);
MSE  = bytes;
Nvox = zeros(T,1);
time = Nvox;

for frame =1:T
    tic;

    % J: depth of octree (for 'MVUB' and '8iVFBv2', usually 9-10)
    [ V,Crgb,J ] = get_pointCloud(dataset, sequence, frame);
    N = size(V,1);
    Nvox(frame) = N;
    C = RGBtoYUV(Crgb); % shape [N x 3]
    
    % ListC: indices of colors
    % FlagsC: indicator of whether a node is a left sibling to another node
    % weightsC: weights of colors
    [ListC,FlagsC,weightsC]=RAHT_param(V,[0 0 0], 2^J, J);
    
    [Coeff,w]=RAHT(C,ListC,FlagsC,weightsC);
    [~,IX_ref] = sort(w,'descend');
    
    Y = Coeff(:,1);

    fprintf('Energy of C: %.4f\n', norm(C));
    fprintf('Energy of Coeff: %.4f\n', norm(Coeff));

    for i=1:nSteps
        %quantize coeffs
        step = colorStep(i);
        Coeff_enc = round(Coeff/step);
        Y_hat = Coeff_enc(:,1)*step;
        U_hat = Coeff_enc(:,2)*step;
        V_hat = Coeff_enc(:,3)*step;
        
        %comptue squared error
        MSE(frame,i) = (norm(Y-Y_hat)^2/(N*255^2));
        
        %encode coeffs sorted according to RAHT weight,  using RLGR
        [nbytesY,xencY]=RLGR_encoder(Coeff_enc(IX_ref,1));
        [nbytesU,xencU]=RLGR_encoder(Coeff_enc(IX_ref,2));
        [nbytesV,xencV]=RLGR_encoder(Coeff_enc(IX_ref,3));
        bytes(frame,i) = nbytesY + nbytesU + nbytesV;
        
    end
    time(frame)=toc;
    disp([time(frame) frame T])
end

 psnr = - 10*log10( mean(MSE,1));
 bpv  = 8*sum(bytes,1)/sum(Nvox);
 plot(bpv,psnr,'b-x');
 grid on;
 axis tight;

folder = sprintf('RA-GFT/results/%s/%s/',dataset,sequence);
mkdir(folder);
filename  = sprintf('%s_RAHT.mat',folder);
save(filename,'MSE','bytes','Nvox','colorStep');


