%% Import data from files
% import age
agefile='C:\Users\Marcelo\Documents\PPMI_Tremor\Age_at_visit.csv';
age=readtable(agefile);
aged=age((strcmp(age.EVENT_ID, 'BL')),:);

% import patient groups
groupsfile='C:\Users\Marcelo\Documents\PPMI_Tremor\Participant_Status.csv';
group=readtable(groupsfile);
% pd are the group 1
pdpatientnumb=group.PATNO(group.COHORT==1);
controlpatientnumb=group.PATNO(group.COHORT==2);
prodopatientnumb=group.PATNO(group.COHORT==4);
pdpatientage=group.ENROLL_AGE(group.COHORT==1);

% import patient demographics
demogfile='C:\Users\Marcelo\Documents\PPMI_Tremor\Demographics.csv';
demog=readtable(demogfile);

% identify UPDRStremor
%Y0
updrsfile='C:\Users\Marcelo\Documents\PPMI_Tremor\MDS_UPDRS_Part_III.csv';
updrs=readtable(updrsfile);
all_table=updrs((strcmp(updrs.EVENT_ID, 'BL') & (strcmp(updrs.PAG_NAME, 'NUPDRS3') | strcmp(updrs.PAG_NAME, 'NUPDR3OF')) & ~strcmp(updrs.PDSTATE, 'ON')),:);
rest_tremor=(all_table.NP3RTARU+all_table.NP3RTALU+all_table.NP3RTARL+all_table.NP3RTALL+all_table.NP3RTALJ);
all_table.resttremorscore=rest_tremor;
all_table.resttremorpres=double(rest_tremor>=1);
all_table.constancy=(all_table.NP3RTCON);
action_tremor=(all_table.NP3PTRMR+all_table.NP3PTRML+all_table.NP3KTRMR+all_table.NP3KTRML);
all_table.actiontremor=action_tremor;
all_table.PD=double(ismember(all_table.PATNO, pdpatientnumb));
all_table.control=double(ismember(all_table.PATNO, controlpatientnumb));
all_table.prodrom=double(ismember(all_table.PATNO, prodopatientnumb));
rig=(all_table.NP3RIGN+all_table.NP3RIGRU+all_table.NP3RIGLU+all_table.NP3RIGRL+all_table.NP3RIGLL);
bk=(all_table.NP3FTAPR+all_table.NP3FTAPL+all_table.NP3HMOVR+all_table.NP3HMOVL+all_table.NP3PRSPR+all_table.NP3PRSPL+...
    all_table.NP3TTAPR+all_table.NP3TTAPL+all_table.NP3LGAGR+all_table.NP3LGAGL+all_table.NP3RISNG+all_table.NP3BRADY);
all_table.rig=rig;
all_table.bk=bk;

%Y1
all_table4=updrs((strcmp(updrs.EVENT_ID, 'V04') & (strcmp(updrs.PAG_NAME, 'NUPDRS3') | strcmp(updrs.PAG_NAME, 'NUPDR3OF')) & ~strcmp(updrs.PDSTATE, 'ON')),:);
rest_tremor4=(all_table4.NP3RTARU+all_table4.NP3RTALU+all_table4.NP3RTARL+all_table4.NP3RTALL+all_table4.NP3RTALJ);
all_table4.resttremorscore=rest_tremor4;
all_table4.resttremorpres=double(rest_tremor4>=1);
all_table4.constancy=(all_table4.NP3RTCON);
action_tremor=(all_table4.NP3PTRMR+all_table4.NP3PTRML+all_table4.NP3KTRMR+all_table4.NP3KTRML);
all_table4.actiontremor=action_tremor;
all_table4.PD=double(ismember(all_table4.PATNO, pdpatientnumb));
all_table4.control=double(ismember(all_table4.PATNO, controlpatientnumb));
all_table4.prodrom=double(ismember(all_table4.PATNO, prodopatientnumb));
rig4=(all_table4.NP3RIGN+all_table4.NP3RIGRU+all_table4.NP3RIGLU+all_table4.NP3RIGRL+all_table4.NP3RIGLL);
bk4=(all_table4.NP3FTAPR+all_table4.NP3FTAPL+all_table4.NP3HMOVR+all_table4.NP3HMOVL+all_table4.NP3PRSPR+all_table4.NP3PRSPL+...
    all_table4.NP3TTAPR+all_table4.NP3TTAPL+all_table4.NP3LGAGR+all_table4.NP3LGAGL+all_table4.NP3RISNG+all_table4.NP3BRADY);
all_table4.rig=rig4;
all_table4.bk=bk4;

%Y2
all_table6=updrs((strcmp(updrs.EVENT_ID, 'V06') & (strcmp(updrs.PAG_NAME, 'NUPDRS3') | strcmp(updrs.PAG_NAME, 'NUPDR3OF')) & ~strcmp(updrs.PDSTATE, 'ON')),:);
rest_tremor6=(all_table6.NP3RTARU+all_table6.NP3RTALU+all_table6.NP3RTARL+all_table6.NP3RTALL+all_table6.NP3RTALJ);
all_table6.resttremorscore=rest_tremor6;
all_table6.resttremorpres=double(rest_tremor6>=1);
all_table6.constancy=(all_table6.NP3RTCON);
action_tremor=(all_table6.NP3PTRMR+all_table6.NP3PTRML+all_table6.NP3KTRMR+all_table6.NP3KTRML);
all_table6.actiontremor=action_tremor;
all_table6.PD=double(ismember(all_table6.PATNO, pdpatientnumb));
all_table6.control=double(ismember(all_table6.PATNO, controlpatientnumb));
all_table6.prodrom=double(ismember(all_table6.PATNO, prodopatientnumb));
rig6=(all_table6.NP3RIGN+all_table6.NP3RIGRU+all_table6.NP3RIGLU+all_table6.NP3RIGRL+all_table6.NP3RIGLL);
bk6=(all_table6.NP3FTAPR+all_table6.NP3FTAPL+all_table6.NP3HMOVR+all_table6.NP3HMOVL+all_table6.NP3PRSPR+all_table6.NP3PRSPL+...
    all_table6.NP3TTAPR+all_table6.NP3TTAPL+all_table6.NP3LGAGR+all_table6.NP3LGAGL+all_table6.NP3RISNG+all_table6.NP3BRADY);
all_table6.rig=rig6;
all_table6.bk=bk6;

%% 
% import DATScan
datscanfile='C:\Users\Marcelo\Documents\PPMI_Tremor\DaTScan_Analysis_13May2023.csv';
dat=readtable(datscanfile);
dattable=dat((strcmp(dat.EVENT_ID, 'SC')) | (strcmp(dat.EVENT_ID, 'U01')), :);
dattableV04=dat((strcmp(dat.EVENT_ID, 'V04')), :);
dattableV06=dat((strcmp(dat.EVENT_ID, 'V06')), :);

Tv = outerjoin(dattable,dattableV04,'keys', "PATNO", 'MergeKeys',true);
Tv = outerjoin(Tv,dattableV06,'keys', "PATNO", 'MergeKeys',true);
Ti = outerjoin(all_table,all_table4,'keys', "PATNO", 'MergeKeys',true);
Tj = outerjoin(Ti,all_table6,'keys', "PATNO", 'MergeKeys',true);
TT = outerjoin(Tv, Tj, 'keys', "PATNO", 'MergeKeys',true);
Tk  =outerjoin(TT, aged, 'keys', "PATNO",  'MergeKeys',true);

PD_tab=Tk(Tk.PD_all_table==1 | Tk.PD_all_table4==1 | Tk.PD==1,:);

righttremor=PD_tab.NP3RTARU+PD_tab.NP3RTARL;
lefttremor=PD_tab.NP3RTALU+PD_tab.NP3RTALL;
jawtremor=PD_tab.NP3RTALJ;

PD_tab.righttremor=righttremor;
PD_tab.lefttremor=lefttremor;
PD_tab.jawtremor=jawtremor;

rt=righttremor>0;
lt=lefttremor>0;
unilattr=((rt+lt)==1);

PD_tab.rt=rt;
PD_tab.lt=lt;
PD_tab.unilattr=unilattr;

writetable(PD_tab, 'allpatclean.xlsx')

%identify patients with and without tremor and write that
ptwithtrem=PD_tab.resttremorpres==1;
ptwithnotrem=PD_tab.resttremorpres==0;

writetable(PD_tab(ptwithtrem,:), 'WithTremor.xlsx')
writetable(PD_tab(ptwithnotrem,:), 'WithNoTremor.xlsx')

new_tab=outerjoin(Tk(Tk.PD_all_table==1 | Tk.PD_all_table4==1 | Tk.PD==1,:), demog, 'keys', "PATNO",  'MergeKeys',true);

writetable(new_tab(ptwithtrem,:), 'WithTremordemog.xlsx')
writetable(new_tab(ptwithnotrem,:), 'WithNoTremordemog.xlsx')

%% Identify the ON & OFF at 2 years (Fig 6)

%Y2 in OFF
all_table6=updrs((strcmp(updrs.EVENT_ID, 'V06') & (strcmp(updrs.PAG_NAME, 'NUPDRS3') | strcmp(updrs.PAG_NAME, 'NUPDR3OF')) & ~strcmp(updrs.PDSTATE, 'ON')),:);
rest_tremor6=(all_table6.NP3RTARU+all_table6.NP3RTALU+all_table6.NP3RTARL+all_table6.NP3RTALL+all_table6.NP3RTALJ);
all_table6.resttremorscore=rest_tremor6;
all_table6.resttremorpres=double(rest_tremor6>=1);
all_table6.constancy=(all_table6.NP3RTCON);
action_tremor=(all_table6.NP3PTRMR+all_table6.NP3PTRML+all_table6.NP3KTRMR+all_table6.NP3KTRML);
all_table6.actiontremor=action_tremor;
all_table6.PD=double(ismember(all_table6.PATNO, pdpatientnumb));
all_table6.control=double(ismember(all_table6.PATNO, controlpatientnumb));
all_table6.prodrom=double(ismember(all_table6.PATNO, prodopatientnumb));
rig6=(all_table6.NP3RIGN+all_table6.NP3RIGRU+all_table6.NP3RIGLU+all_table6.NP3RIGRL+all_table6.NP3RIGLL);
bk6=(all_table6.NP3FTAPR+all_table6.NP3FTAPL+all_table6.NP3HMOVR+all_table6.NP3HMOVL+all_table6.NP3PRSPR+all_table6.NP3PRSPL+...
    all_table6.NP3TTAPR+all_table6.NP3TTAPL+all_table6.NP3LGAGR+all_table6.NP3LGAGL+all_table6.NP3RISNG+all_table6.NP3BRADY);
all_table6.rig=rig6;
all_table6.bk=bk6;

%Y2 in ON
all_table6ON=updrs(strcmp(updrs.EVENT_ID, 'V06') & (strcmp(updrs.PAG_NAME, 'NUPDRS3A') | strcmp(updrs.PAG_NAME, 'NUPDR3ON') | strcmp(updrs.PDSTATE, 'ON')),:);
rest_tremor6ON=(all_table6ON.NP3RTARU+all_table6ON.NP3RTALU+all_table6ON.NP3RTARL+all_table6ON.NP3RTALL+all_table6ON.NP3RTALJ);
all_table6ON.resttremorscore=rest_tremor6ON;
all_table6ON.resttremorpres=double(rest_tremor6ON>=1);
all_table6ON.constancy=(all_table6ON.NP3RTCON);
action_tremorON=(all_table6ON.NP3PTRMR+all_table6ON.NP3PTRML+all_table6ON.NP3KTRMR+all_table6ON.NP3KTRML);
all_table6ON.actiontremor=action_tremorON;
all_table6ON.PD=double(ismember(all_table6ON.PATNO, pdpatientnumb));
all_table6ON.control=double(ismember(all_table6ON.PATNO, controlpatientnumb));
all_table6ON.prodrom=double(ismember(all_table6ON.PATNO, prodopatientnumb));
rig6ON=(all_table6ON.NP3RIGN+all_table6ON.NP3RIGRU+all_table6ON.NP3RIGLU+all_table6ON.NP3RIGRL+all_table6ON.NP3RIGLL);
bk6ON=(all_table6ON.NP3FTAPR+all_table6ON.NP3FTAPL+all_table6ON.NP3HMOVR+all_table6ON.NP3HMOVL+all_table6ON.NP3PRSPR+all_table6ON.NP3PRSPL+...
    all_table6ON.NP3TTAPR+all_table6ON.NP3TTAPL+all_table6ON.NP3LGAGR+all_table6ON.NP3LGAGL+all_table6ON.NP3RISNG+all_table6ON.NP3BRADY);
all_table6ON.rig=rig6ON;
all_table6ON.bk=bk6ON;
tremrightON=all_table6ON.NP3RTARU+all_table6ON.NP3RTARL;
tremleftON=all_table6ON.NP3RTALU+all_table6ON.NP3RTALL;
bkrightON=(all_table6ON.NP3FTAPR+all_table6ON.NP3HMOVR+all_table6ON.NP3PRSPR+...
    all_table6ON.NP3TTAPR+all_table6ON.NP3LGAGR);
bkleftON=(all_table6ON.NP3FTAPL+all_table6ON.NP3HMOVL+all_table6ON.NP3PRSPL+...
    all_table6ON.NP3TTAPL+all_table6ON.NP3LGAGL);
all_table6ON.tremright=tremrightON;
all_table6ON.tremleft=tremleftON;
all_table6ON.bkright=bkrightON;
all_table6ON.bkleft=bkleftON;

% Fuse and classify the types of response
TableFused = outerjoin(all_table6ON(all_table6ON.PD==1,:),all_table6(all_table6.PD==1,:), 'keys', "PATNO", 'MergeKeys',true);

TableFused.changetremor=((TableFused.resttremorscore_left-TableFused.resttremorscore_right)./TableFused.resttremorscore_right);
TableFused.changebk=((TableFused.bk_left-TableFused.bk_right)./TableFused.bk_right);
TableFused.changer=((TableFused.rig_left-TableFused.rig_right)./TableFused.rig_right);

TableFused.resolution=((TableFused.resttremorscore_left-TableFused.resttremorscore_right)./TableFused.resttremorscore_right)==-1;
TableFused.improvement=(((TableFused.resttremorscore_left-TableFused.resttremorscore_right)./TableFused.resttremorscore_right)>-1 &...
    ((TableFused.resttremorscore_left-TableFused.resttremorscore_right)./TableFused.resttremorscore_right)<0);
TableFused.equal=((TableFused.resttremorscore_left-TableFused.resttremorscore_right)./TableFused.resttremorscore_right)==0;
TableFused.worsen=((TableFused.resttremorscore_left-TableFused.resttremorscore_right)./TableFused.resttremorscore_right)>0;

writetable(TableFused,'ChangeTables.xlsx')

Tv(ismember(Tv.PATNO, TableFused.PATNO(TableFused.resttremorpres_right==1 & TableFused.PD_right==1)),:)
Fusion2 = outerjoin(Tv(ismember(Tv.PATNO, TableFused.PATNO(TableFused.resttremorpres_right==0 & TableFused.PD_right==1)),:),TableFused,'keys', "PATNO", 'MergeKeys',true);

FusionPD=Fusion2;
NewFusion=FusionPD(FusionPD.resttremorpres_right==0,:);

NewFusion.resttremorscore_right-NewFusion.resttremorscore_left==0;
NewFusion(NewFusion.resttremorscore_right-NewFusion.resttremorscore_left<0,:)

writetable(NewFusion,'ChangeWithDemogandDAT.xlsx')
%% Perform simulations (Figure 5)

indexes=0:0.005:0.5;

% more complex simulation with change of the 2 variables
npt=1
for jt=1:101
    CDt = normrnd(mean(nanmean(xt(xt(:,3)==1,1:2)))-indexes(jt),mean(nanstd(xt(xt(:,3)==1,1:2))),[4000,1]);
    [x,y]=corrcoef(xt(xt(:,3)==1,1), xt(xt(:,3)==1,2), 'rows', 'complete');
    CEt = randwithcorr(CDt,x(2),mean(nanmean(xt(xt(:,3)==1,1:2))),mean(nanstd(xt(xt(:,3)==1,1:2))));

    CDnt = normrnd(mean(nanmean(xt(xt(:,3)==0,1:2))),mean(nanstd(xt(xt(:,3)==0,1:2))),[6000,1]);
    [x,y]=corrcoef(xt(xt(:,3)==0,1), xt(xt(:,3)==0,2), 'rows', 'complete');
    CEnt = randwithcorr(CDnt,x(2),mean(nanmean(xt(xt(:,3)==0,1:2)))-indexes(jt)/2,mean(nanstd(xt(xt(:,3)==0,1:2))));

    CEsq=abs(vertcat(CEt, CEnt));
    CDir=abs(vertcat(CDt, CDnt));

    onies=1*ones(floor(sum(xt(:,9)==1)./sum(xt(:,9)>0)*4000), 1);
    for a=2:6
        onies=vertcat(onies, a*ones(floor(sum(xt(:,9)==a)./sum(xt(:,9)>0)*4000), 1));
    end
    u3=vertcat(onies, ones(4000-numel(onies),1));
    v3=zeros(size(CEnt,1),1);
    TremE=vertcat(u3, v3);

    [xD,yD]=corrcoef(CDir, TremE);

    [xE,yE]=corrcoef(CEsq, TremE);
% 
% subplot(2,1,1)
% scatter(CDir, TremE, 12, 'filled')
% lsline
% xlim([0 4])
% xlabel('CBP Contra')
% ylabel('Rest Tremor')
% 
% subplot(2,1,2)
% scatter(CEsq, TremE, 12, 'filled')
% lsline
% xlim([0 4])
% xlabel('CBP Ipsi')
% ylabel('Rest Tremor')
% 
% x0=10;
% y0=10;
% width=200;
% height=200
% set(gcf,'position',[x0,y0,width,height])

    correlations(npt,1)=xD(2);
    correlations(npt,2)=xE(2);
    npt=npt+1;
end

% plots
plot(0:0.005:0.5, correlations)
xlabel('CBP Asymmetry (Contralateral reduced)')
ylabel('Pearson r')
legend('Contra', 'Ipsi')
x0=10;
y0=10;
width=270;
height=200;
box off
set(gcf,'position',[x0,y0,width,height])

CEsq=abs(vertcat(CEt, CEnt));
CDir=abs(vertcat(CDt, CDnt));

[CIt, CIt2]=histcounts(abs(CEt), [0:0.25:4], 'Normalization', 'probability');
[CCt, CCt2]=histcounts(abs(CDt), [0:0.25:4], 'Normalization', 'probability');
[CInt, CInt2]=histcounts(abs(CEnt), [0:0.25:4], 'Normalization', 'probability');
[CCInt, CCInt2]=histcounts(abs(CDnt), [0:0.25:4], 'Normalization', 'probability');

subplot(2,2,1), 
plot(CCt2(1:end), horzcat(0,CCt))
xticks(mean(CDt))
xlim([0 4])
ylim([0 0.2])
subplot(2,2,2), 
plot(CCt2(1:end), horzcat(0,CIt))
xticks(mean(CEt))
xlim([0 4])
ylim([0 0.2])
subplot(2,2,3), 
plot(CCt2(1:end), horzcat(0,CCInt))
xticks(mean(CDnt))
xlim([0 4])
ylim([0 0.2])
subplot(2,2,4), 
plot(CCt2(1:end), horzcat(0,CInt))
xticks(mean(CEnt))
xlim([0 4])
ylim([0 0.2])

x0=10;
y0=10;
width=270;
height=200;
set(gcf,'position',[x0,y0,width,height])