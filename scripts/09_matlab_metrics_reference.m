clear;
clc;


%% =========================================================
% Disease27Edge Dynamic Metrics
%
% FCM_21ROI
% Disease 27 edges
% K-means K=2
%
%% =========================================================


%% ===============================
% Load cluster result
%% ===============================


clusterFile = '<PROJECT_ROOT>/DBC/FCM_21ROI/GCM_Disease27Edge/mat_correlation_Kmeans_2/cluster_index.mat';


load(clusterFile,...
    'IDX_subj',...
    'K',...
    'Subindex',...
    'IDXall',...
    'SubFold');



fprintf('\n==============================\n');
fprintf('Disease27Edge Dynamic Analysis\n');
fprintf('==============================\n');

fprintf('K = %d\n',K);



%% ===============================
% Subject information
%% ===============================


Subjects = string(SubFold);

nSub = length(Subjects);


fprintf('Subjects = %d\n',nSub);



%% ===============================
% Group classification
%% ===============================


Group = strings(nSub,1);



for i=1:nSub


    id = Subjects(i);


    if startsWith(id,"H")

        Group(i)="HC";


    elseif contains(id,"M0")

        Group(i)="M0";


    elseif contains(id,"M1")

        Group(i)="M1";


    else

        Group(i)="Unknown";

    end


end



disp('Group counts:')

tabulate(Group)



%% ===============================
% Initialize
%% ===============================


State1_FO=zeros(nSub,1);
State2_FO=zeros(nSub,1);


State1_MDT=zeros(nSub,1);
State2_MDT=zeros(nSub,1);


P11=zeros(nSub,1);
P12=zeros(nSub,1);
P21=zeros(nSub,1);
P22=zeros(nSub,1);


NT=zeros(nSub,1);

SwitchRate=zeros(nSub,1);



%% ===============================
% Calculate metrics
%% ===============================


for s=1:nSub


    %% get state sequence

    if iscell(IDX_subj)

        states = IDX_subj{s};

    else

        states = IDX_subj(s,:);

    end


    states = states(~isnan(states));


    nWin = length(states);



    %% FO

    State1_FO(s)=sum(states==1)/nWin;

    State2_FO(s)=sum(states==2)/nWin;



    %% transitions

    trans = states(2:end) ~= states(1:end-1);


    NT(s)=sum(trans);

    SwitchRate(s)=NT(s)/(nWin-1);



    %% dwell time


    change=[1 find(trans)+1 nWin+1];


    stateList=[];
    lengthList=[];



    for k=1:length(change)-1


        a=change(k);

        b=change(k+1)-1;


        stateList(end+1,1)=states(a);

        lengthList(end+1,1)=b-a+1;


    end



    if any(stateList==1)

        State1_MDT(s)=mean(lengthList(stateList==1));

    end


    if any(stateList==2)

        State2_MDT(s)=mean(lengthList(stateList==2));

    end



    %% transition probability


    n11=0;
    n12=0;
    n21=0;
    n22=0;



    for k=1:nWin-1


        if states(k)==1 && states(k+1)==1

            n11=n11+1;

        elseif states(k)==1 && states(k+1)==2

            n12=n12+1;

        elseif states(k)==2 && states(k+1)==1

            n21=n21+1;

        elseif states(k)==2 && states(k+1)==2

            n22=n22+1;

        end


    end



    if n11+n12>0

        P11(s)=n11/(n11+n12);

        P12(s)=n12/(n11+n12);

    end



    if n21+n22>0

        P21(s)=n21/(n21+n22);

        P22(s)=n22/(n21+n22);

    end


end



%% ===============================
% Subject result table
%% ===============================


Result=table(...
    Subjects,...
    Group,...
    State1_FO,...
    State2_FO,...
    State1_MDT,...
    State2_MDT,...
    P11,...
    P12,...
    P21,...
    P22,...
    NT,...
    SwitchRate);



outDir='<PROJECT_ROOT>/DBC/FCM_21ROI/Disease27Edge_Statistics';



if ~exist(outDir,'dir')

    mkdir(outDir);

end



writetable(Result,...
    fullfile(outDir,...
    'Disease27Edge_K2_AllSubjects.csv'));



%% ===============================
% Statistics
%% ===============================


MetricName={...
'State1_FO';
'State2_FO';
'State1_MDT';
'State2_MDT';
'P11';
'P12';
'P21';
'P22';
'NT';
'SwitchRate'
};



Data={...
State1_FO;
State2_FO;
State1_MDT;
State2_MDT;
P11;
P12;
P21;
P22;
NT;
SwitchRate
};



Metric=[];
t_M0HC=[];
p_M0HC=[];
d_M0HC=[];


t_M1M0=[];
p_M1M0=[];
d_M1M0=[];



for m=1:length(MetricName)


    x=Data{m};


    HC=x(Group=="HC");

    M0=x(Group=="M0");

    M1=x(Group=="M1");



    %% M0 vs HC


    [~,p1,st1]=ttest2(M0,HC,...
        'Vartype','unequal');


    t1=st1.tstat;



    d1=(mean(M0)-mean(HC))/...
        sqrt((std(M0)^2+std(HC)^2)/2);



    %% M1 vs M0


    [~,p2,st2]=ttest(M1,M0);


    t2=st2.tstat;


    d2=mean(M1-M0)/std(M1-M0);



    Metric=[Metric;MetricName(m)];


    t_M0HC=[t_M0HC;t1];

    p_M0HC=[p_M0HC;p1];

    d_M0HC=[d_M0HC;d1];



    t_M1M0=[t_M1M0;t2];

    p_M1M0=[p_M1M0;p2];

    d_M1M0=[d_M1M0;d2];


end



Statistics=table(...
    Metric,...
    t_M0HC,...
    p_M0HC,...
    d_M0HC,...
    t_M1M0,...
    p_M1M0,...
    d_M1M0);



writetable(Statistics,...
    fullfile(outDir,...
    'Disease27Edge_K2_Statistics.csv'));



disp('==============================')
disp('COMPLETE')
disp('==============================')


disp(Statistics)
