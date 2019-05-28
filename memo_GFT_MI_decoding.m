% The following Matlab code implements the GFT-decoder for MI decoding
% as described in "Connectivity steered Graph Fourier Transform for Motor
% Imagery BCI Decoding" by K Georgiadis, N Laskaris, S Nikolopoulos and I
% Kompatsiaris published in Journal of Neural Engineering.
% https://doi.org/10.1088/1741-2552/ab21fd

clear;
load SMR_data_examplar.mat
XXtrials=permute(SMR_trials,[2 1 3]);
        
t_start=3;t_end=7; % define the starting and ending point for each trial
Fbands=[1 4; 4 7.5; 8 10; 10 13; 13 20; 20 30; 30 45]; % Define the frequency bands
   
% Build the Wmulti matrix based on the CFC estimations
MultiRows_ALL=[];
TRIAL_CORR=[];for i_trial=1:size(XXtrials,1),%i_trial
   Trial_DATA=squeeze(XXtrials(i_trial,:,t_start*Fs+1:t_end*Fs));
   for i=1:size(Fbands,1); [b2,a2]=butter(3,[Fbands(i,1)/(Fs/2),Fbands(i,2)/(Fs/2)]);Filt_b(i,:)= b2; Filt_a(i,:)=a2; end
   f_Trial_DATA=[];for i_freq=1:size(Fbands,1);f_Trial_DATA(:,:,i_freq)=filtfilt(Filt_b(i_freq,:),Filt_a(i_freq,:),Trial_DATA')';end 
 MultiRows=[]; for i_sensor=1:size(XXtrials,2), rows=squeeze(f_Trial_DATA(i_sensor,:,:))'; MultiRows=[MultiRows;rows]; end 
 MultiEnvelopes=abs(hilbert(MultiRows'))'; TrialCORR=abs(corr(MultiEnvelopes')); TrialCORR=TrialCORR-diag(diag(TrialCORR)); % Equation #1
 MultiEnvelopes_ALL(i,:,:)=MultiEnvelopes;
 TRIAL_CORR(i_trial,:,:)=TrialCORR;
 MultiRows_ALL(i_trial,:,:) = MultiRows;
end   

% Select the trials corresponding to one of the two recording conditions
%to build the GFT base
EC_AA = TRIAL_CORR(1:numel(find(SMR_labels==1)),:,:);

% Formulate the GFT base while incorporating the LOOCV scheme, so as to
% exclude the test-trial from the base 
V_A_ALL=[];
for i=1:size(EC_AA,1)
    A_sub = cat(1,EC_AA(1:i-1,:,:),EC_AA(i+1:end,:,:));
    EC_A_avg = squeeze(mean(A_sub,1));
    EC_A_avg_ALL(i,:,:)=EC_A_avg;
    spA=EC_A_avg;
    L_A=diag(sum(spA))-spA; I_A=diag(ones(1,size(SMR_trials,1)));
    [V_A,Lamda_A]=eig(L_A);
    V_A_ALL(i,:,:)=V_A;
end
EC_A_avg=squeeze(mean(EC_AA,1));
EC_A_avg_ALL(size(EC_AA,1)+1,:,:)=EC_A_avg;
spA=EC_A_avg;
L_A=diag(sum(spA))-spA; I_A=diag(ones(1,size(SMR_trials,1)));
[V_A,Lamda_A]=eig(L_A);
V_A_ALL(size(EC_AA,1)+1,:,:) =V_A;

% Power Estimation
Energies=[];
for kk=1:size(TRIAL_CORR,1)
    ST=squeeze(MultiRows_ALL(kk,:,:));
    if kk<=numel(find(SMR_labels==1))
        FourierSTs = squeeze(V_A_ALL(kk,:,:))'*ST; % Equation #2
    else
        FourierSTs = squeeze(V_A_ALL(numel(find(SMR_labels==1))+1,:,:))'*ST; % Equation #2
    end
    trial_Energy=sum(abs(FourierSTs')); % Equation #4
    Energies(kk,:)=trial_Energy;
end

GROUP=SMR_labels;
FVs=Energies';

% Linear SVM for the classification task using the LOOCV scheme
predict_labels=[];Score2_ALL=[];
for i=1:size(FVs,2)
    train_set=[];train_labels=[];
    train_set = cat(2,FVs(:,1:i-1),FVs(:,i+1:end));
    train_labels = cat(1,GROUP(1:i-1),GROUP(i+1:end));
    %threshold definition process
    scores_perm=[];
    train_labels_rand_perm=[];
    np=11; %define the number of permutation for the threshold estimation
    for i_perm=1:np
        Score2_perm=[];
        r_idx=randperm(size(train_labels,1));
        train_labels_rand_perm(r_idx,1)=train_labels(:,1);
        [IDX_perm,Score2_perm]=rankfeatures(train_set,train_labels_rand_perm,'criterion','wilcoxon');
        scores_perm(i_perm,:)=Score2_perm;
    end
    scores_perm_med=median(scores_perm);
    %define the threshold rule using either option #1 or option #2
    thress = prctile(scores_perm_med,95); % option #1
%     thress = mean(scores_perm_med)+3*std(scores_perm_med); % option #2
    [IDX,Score2]=rankfeatures(train_set,train_labels,'criterion','wilcoxon');
     Mdl = fitcsvm(train_set(IDX(1:numel(find(Score2>thress))),:)',train_labels','Standardize',true,'KernelFunction','linear','KernelScale','auto');%
    test=FVs(IDX(1:numel(find(Score2>thress))),i);
    [l,s]=predict(Mdl,test');
    predict_labels(i)=l;
end
Classification_Error=numel(find(predict_labels'-GROUP))/size(GROUP,1);
