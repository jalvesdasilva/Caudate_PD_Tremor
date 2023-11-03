# -*- coding: utf-8 -*-
"""
@authors: 
Pedro Ferreira
Joaquim A. da Silva
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import cos, sin, pi
from scipy import signal
import glob
import imufusion
import seaborn as sns

def rotator(df):
    """
    Rotator function takes in triaxial total acceleration, converts into 
    ground and body acceleration components, computes Euler angles and rotates
    the sensor to a predefined orientation.
    """
    
    acc_freq = 40
    fs = acc_freq  # Sampling frequency
    fc = 1  # Cut-off frequency of the filter
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(5, w, 'low')

    #define input total acceleration into body and ground acceleration
    BaccelX=df['AccX']-signal.filtfilt(b, a, df['AccX'])
    BaccelY=df['AccY']-signal.filtfilt(b, a, df['AccY'])
    BaccelZ=df['AccZ']-signal.filtfilt(b, a, df['AccZ'])
    GaccelX=signal.filtfilt(b, a, df['AccX'])
    GaccelY=signal.filtfilt(b, a, df['AccY'])
    GaccelZ=signal.filtfilt(b, a, df['AccZ'])

    df['BaccelX'] = BaccelX
    df['BaccelY'] = BaccelY
    df['BaccelZ'] = BaccelZ

    df['GaccelX'] = GaccelX
    df['GaccelY'] = GaccelY
    df['GaccelZ'] = GaccelZ

    """rotations"""
    
    #x, y and z matrices
    def R_x(x):
        return np.array([[1,      0,       0],
                         [0,cos(x),-sin(x)],
                         [0,sin(x), cos(x)]])

    def R_y(y):
        return np.array([[cos(y),0,sin(y)],
                         [0,      1,        0],
                         [-sin(y), 0, cos(y)]])

    def R_z(z):
        return np.array([[cos(z),-sin(z),0],
                         [sin(z), cos(z),0],
                         [0,      0,       1]])
    
    #define arrays of total, body and ground acceleration
    accel = np.array([df['AccX'],
                      df['AccY'],
                      df['AccZ']])
    grav = np.array([df['GaccelX'],
                     df['GaccelY'],
                     df['GaccelZ']])
    body = np.array([df['BaccelX'],
                     df['BaccelY'],
                     df['BaccelZ']])
    timestamp = df.PacketCounter.values
    gyroscope = df.iloc[:,7:10].values
    accelerometer = df.iloc[:,4:7].values
    
    # use the imufusion package to calculate euler angles
    ahrs = imufusion.Ahrs()
    euler = np.empty((len(timestamp), 3))
    for index in range(len(timestamp)):
        ahrs.update_no_magnetometer(gyroscope[index], accelerometer[index], 1 / 40)
        euler[index] = ahrs.quaternion.to_euler()

    pitch = euler[:, 1]*pi/180
    roll = euler[:, 0]*pi/180
    yaw = euler[:, 2]*pi/180
    
    # pitch = df['Pitch']
    # roll = df['Roll']
    # yaw = df['Yaw']

    earth_accels = np.empty(accel.shape)
    earth_gravity = np.empty(accel.shape)
    
    #rotate to the earth reference
    for i in range(df.shape[0]):
        earth_accels[:,i] = R_z(yaw[i]) @ R_y(pitch[i]) @ R_x(roll[i]) @ accel[:,i]
        earth_gravity[:,i] = R_z(yaw[i]) @ R_y(pitch[i]) @ R_x(roll[i]) @ grav[:,i]
    earth_body = earth_accels-earth_gravity
    
    #create dataframe columns from the outputs of earth reference rotation 
    df['rotated_AccX'] = earth_accels[0,:]
    df['rotated_AccY'] = earth_accels[1,:]
    df['rotated_AccZ'] = earth_accels[2,:]
    df['rotated_GaccelX'] = earth_gravity[0,:]
    df['rotated_GaccelY'] = earth_gravity[1,:]
    df['rotated_GaccelZ'] = earth_gravity[2,:]
    df['rotated_BaccelX'] = earth_body[0,:]
    df['rotated_BaccelY'] = earth_body[1,:]
    df['rotated_BaccelZ'] = earth_body[2,:]
    
    return df


def welchFunc(data,fs):
    """
    welchFunc receives a timeseries and estimates the Welch power spectral
    density.
    """
    try:
        #input data
        x = data
        #estimate power spectral density (PSD)
        fP, Pxx = signal.welch(x, fs = fs)

        #normalize PSD to the mean power between 1-3Hz
        freq_for_norm = np.mean(Pxx[(fP>1) & (fP<3)])
        dataNormWelch = Pxx / freq_for_norm
        
    except Exception as e:
        print(e)
        dataNormWelch = np.nan
        fP= np.nan

    return(dataNormWelch, fP)


def builder(path, body_segment, acc_freq, low_f, high_f, norm_type):
    """
    Builder function is used to iteratively process individual inertial sensor 
    data and builds a dataframe of subjects. Function takes in the following
    parameters:
        path - folder directory
        body_segment - sufix referring to the body segment of interest
        acc_freq - accelerometer sampling frequency
        low_f - lower boundary of frequency
        high_f - higher boundary of frequency
        norm_type - function to be called for PSD estimation
    """
    from sklearn.decomposition import PCA
    
    #define sampling frequency and labels for input acceleration data
    acc_freq = acc_freq
    axis_X = 'rotated_AccX' #in this case, we are reading the rotated axis X
    axis_Y = 'rotated_AccY' #Y
    axis_Z = 'rotated_AccZ' #Z
    
    #these are the boundaries of frequency
    low_f=low_f 
    high_f=high_f

    IDs = []

    #define a set of lists to which data will be iteratively added
    x_welch_freq=[]
    
    x_welch_ratio=[]

    f_all=[]

    T_accel=[]

    rotated_AccX = []
    rotated_AccY = []
    rotated_AccZ = []
    
    PC_out = []

    dfs = []

    path_res = path
    
    #recursively look for all files that have the body segment sufix within the given directory path
    res = [file for file in glob.glob(path_res + '/*{}*.txt'.format(body_segment), recursive=True)]
    
    #loop that iteratively processes individual data                                   
    print (len(res))                                                                                                           
    for file in res:
        print(file)
        #set ID of subject based on folder name
        ID = '\\'.join(file.split('\\')[10:-3])
        df_in = pd.read_csv(file, sep='\t', comment='/')
        
        IDs.append(ID)
        
        #rotate raw signals
        df = rotator(df_in)
        #first and last 100 data points are trimmed to exclude rotation artifacts
        df = df.iloc[100:-100]
        
        #perform PCA on the 3 acceleration axis
        model = PCA(n_components=1)

        fs = acc_freq  # Sampling frequency
        fc = 1  # Cut-off frequency of the filter
        w = fc / (fs / 2) # Normalize the frequency
        b, a = signal.butter(5, w, 'low')
        BaccelX=df[axis_X]-signal.filtfilt(b, a, df[axis_X])
        BaccelY=df[axis_Y]-signal.filtfilt(b, a, df[axis_Y])
        BaccelZ=df[axis_Z]-signal.filtfilt(b, a, df[axis_Z])
        X = pd.DataFrame({'BaccelX':BaccelX, 'BaccelY':BaccelY, 'BaccelZ':BaccelZ})
        features_list = [x for x in X[['BaccelX', 'BaccelY', 'BaccelZ']]]
        x_PCA = X.loc[:, features_list].values
        #new PCA transformed axis saved as PC0
        df['PC0'] = model.fit_transform(x_PCA)
        
        #Total acceleration, saved but not used in the analysis
        Total=np.sqrt(BaccelX**2+BaccelY**2+BaccelZ**2)
        
        #run welchFunc in PC0
        X_freq_welch, f=norm_type(df['PC0'],fs)
        #extract maximum power between the predefined frequency boundaries
        ratio_X_welch=np.max(X_freq_welch[(f>low_f) & (f<high_f)])

        #append variables to lists set before
        x_welch_freq.append(X_freq_welch)
        x_welch_ratio.append(ratio_X_welch)
        
        T_accel.append(Total)
        
        rotated_AccX.append(df['rotated_AccX'].values)
        rotated_AccY.append(df['rotated_AccY'].values)
        rotated_AccZ.append(df['rotated_AccZ'].values)
        
        PC_out.append(df['PC0'])
        f_all.append(f)
        
        dfs.append(df)
        
    #setting dataframe to be returned from builder
    df_out = pd.DataFrame()

    df_out['rotated_AccX'] = rotated_AccX
    df_out['rotated_AccY'] = rotated_AccY
    df_out['rotated_AccZ'] = rotated_AccZ
    df_out['X_freq_welch']= x_welch_freq
    df_out['X_ratio_welch']=x_welch_ratio
    df_out['f_bins']=f_all

    df_out['T_accel']=T_accel
    
    df_out['PC0'] = PC_out

    df_out['IDs'] = IDs
    df_out.set_index('IDs', drop=True, inplace=True)

    return df_out

def mean_data_per_human(data,var):
    """
    Compute mean data per participant.
    """
    humans=np.unique(data.index)
    data_all=[]
    for human in humans:
        temp=data[data.index==human]
        dat=np.nanmean(np.vstack(temp[var].values),axis=0)
        data_all.append(dat)
    data_all=np.array(data_all)

    return(data_all)

def plots_with_shaded_error(xdata,ydata,error,color,error_color,alfa,label):
    """
    Auxiliary function for plotting timseries with shaded errors.
    """
    ax=plt.gca()
    ax.fill_between(xdata, ydata-error, ydata+error,facecolor=error_color,alpha=alfa)
    plt.plot(xdata,ydata,color,label=label)

#%%

def plot_stats(data, x, y, hue):
    """
    Auxiliary function for plotting correlations and bar plots.
    """
    from scipy import stats
    from statannot import add_stat_annotation
    #x and y variables of interest
    xoi = x
    yoi = y
    plt.figure()
    zg = {'Positive': 1.1, 'Negative': 1.18}
    plt.xlim(0, 4)
    
    #first correlation plot with groups separated based on hue variable
    g = sns.lmplot(x='{}'.format(xoi), y='{}'.format(yoi), data=data, hue='group', 
                   legend_out='False', palette=['red', 'black'], ci=None,
                   scatter_kws={"s": 100, 'alpha':0.3})
    #g.set(yticks=(0, 5, 10, 12)) #optinally set y axis ticks manually
    #g.set(xticks=(0, 2, 4, 6, 8)) #optinally set x axis ticks manually
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    
    #second correlation plot for the whole cohort
    g_all = sns.lmplot(x='{}'.format(xoi), y='{}'.format(yoi), data=data, 
                       legend_out='False', ci=None, scatter_kws={"s": 100, 'alpha':0.3})
    
    for ax in g.axes.flatten():
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='x')
    for ax in g_all.axes.flatten():
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='x')

    #function for annotating correlation stats separated by group. Stats are 
    #computed by scipy.stats package
    def annotate(data, **kws):
        g = data.group.unique()[0]
        z = zg[g]
        x, y = data['{}'.format(xoi)], data['{}'.format(yoi)]
        nas=np.logical_or(np.isnan(x), np.isnan(y)) #prepares data in case of NaNs
        r, p = stats.pearsonr(x[~nas], y[~nas])
        ax = plt.gca()
        ax.text(0.05, z, f'{g}: r={r:.2f}, p={p:.2f}', transform=ax.transAxes)
    
    #function for annotating correlation stats for whole cohort. Stats are 
    #computed by scipy.stats package
    def annotate_all(data, **kws):
        x, y = data['{}'.format(xoi)], data['{}'.format(yoi)]
        nas=np.logical_or(np.isnan(x), np.isnan(y)) #prepares data in case of NaNs
        r, p = stats.pearsonr(x[~nas], y[~nas])
        ax = plt.gca()
        ax.text(0.05, 1, 'Combined: r={:.2f}, p={:.2f}'.format(r, p), transform=ax.transAxes)
        
    #annotates correlation stats
    _ = g.map_dataframe(annotate)
    plt.savefig('correlation.svg', format='svg')
    _ = g_all.map_dataframe(annotate_all)
    
    sns.set_context("talk", rc={"axes.labelsize":10})
    sns.set_context("talk", font_scale=0.8)

    #sworm and bar plots on the same figure. Stats are computed by statannot package
    plt.figure(figsize=(2.5, 5))
    swarm = sns.swarmplot(data=data, x='{}'.format(hue), y='{}'.format(yoi), 
                          palette=['red', 'grey'])
    sns.barplot(x='{}'.format(hue), y='{}'.format(yoi), data=data, capsize=0.1, 
                alpha=0.5, errwidth=2, ax=swarm, palette=['red', 'grey'])
    add_stat_annotation(swarm, data=data, x='{}'.format(hue), y='{}'.format(yoi), 
                        box_pairs=[('Positive', 'Negative')], test='t-test_ind', 
                        text_format='star', loc='outside')
    swarm.set(xlabel=None)
    plt.ylabel('{}'.format(yoi))
    plt.locator_params(axis='y', nbins=5)
    plt.yticks(fontsize=20)
    plt.show()
    
#%%
"""
Run data processing for positive group (posture trial)
"""
plt.rcParams["font.family"] = "Verdana"
acc_freq = 40 #set sampling frequency
#Path for the positive group. Structure allows for iterative search
path_positive = r'C:\Users\Admin\Desktop\Pedro\CCU\Dados - Cópia\DaTscan\datscan_with_files_matched\positive\**\balance\ex0'
#Set body segment to be processed
body_segment = 'lower_back'

#Run data processing for the positive group, data is saved as a df_positive
df_positive = builder(path=path_positive, body_segment=body_segment, 
                      acc_freq=acc_freq, low_f=4, high_f=6, norm_type=welchFunc)

#%%
"""
Run data processing for negative group (posture trial)
"""
acc_freq = 40 #prepares data in case of NaNs
#Path for the negative group. Structure allows for iterative search
path_negative = r'C:\Users\Admin\Desktop\Pedro\CCU\Dados - Cópia\DaTscan\datscan_with_files_matched\negative\**\balance\ex0'
#Set body segment to be processed
body_segment = 'lower_back'

#Run data processing for the negative group, data is saved as a df_negative
df_negative = builder(path=path_negative, body_segment=body_segment, 
                      acc_freq=acc_freq, low_f=4, high_f=6, norm_type=welchFunc)

#%%
"""
Plot comparing Welch PSD between postive and negative groups
"""
plt.rcdefaults()
plt.rcParams["font.family"] = "Verdana"

low_f=4
high_f=6
f_datscan = df_positive.f_bins[0]

plt.figure(2, figsize=(5, 5))
X_freq_welch_positive = mean_data_per_human(df_positive,'X_freq_welch')
X_freq_welch_negative = mean_data_per_human(df_negative,'X_freq_welch')
#Log transform welch timeseries
log_welch_positive = np.log(X_freq_welch_positive)
log_welch_negative = np.log(X_freq_welch_negative)

plots_with_shaded_error(f_datscan,np.mean(log_welch_positive,axis=0),
                        np.std(log_welch_positive,axis=0)/np.sqrt(log_welch_positive.shape[0]),
                        'red','red',0.5,'Positive (n={})'.format(len(df_positive)))
plots_with_shaded_error(f_datscan,np.mean(log_welch_negative,axis=0),
                        np.std(log_welch_negative,axis=0)/np.sqrt(log_welch_negative.shape[0]),
                        'k','k',0.5,'Negative (n={})'.format(len(df_negative)))
plt.xlim([1,10])
plt.ylim([-1,2])
plt.legend(fontsize=12)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.locator_params(axis='x', nbins=5)
plt.locator_params(axis='y', nbins=3)
plt.ylabel('Power (normalized)', fontsize=15)
plt.xlabel('Hz',  fontsize=15)
plt.tight_layout()
plt.show()

#%%
"""
Merge kinematic dataframes with clinical and imaging (DaTscan) data
"""
#Join positive and negative dataframes
df_kinematics = pd.DataFrame()
df_kinematics = pd.concat([df_positive, df_negative])

#Path and read clinical data
path_clinical = r'C:\Users\Admin\Desktop\Pedro\CCU\PD-DaTscan\demographics_CLEAN_V3.csv'
df_clinical = pd.read_csv(path_clinical, sep=';', index_col='ID')

df_clinical.drop(['MCO947', 'JFI959'], inplace=True) #MCO947 and JFI959 have no kinematic data
df_clinical = df_clinical[df_clinical.group != 'CRF VAZIO???'] #drop those who have no clinical data

#Compute new UPDRS metrics from separate UPDRS items
#Total tremor
df_clinical['Tremor'] = df_clinical[['3. 15 D', '3.15 E', '3.16 D', '3. 16 E', 
                                     '3.17 MSD', '3.17 MSE', '3.17 MID', 
                                     '3.17 MIE', '3.17 jaw', '3.18']].sum(axis=1)
#PIGD
df_clinical['PIGD'] = df_clinical[['3.10', '3.11', '3.12']].sum(axis=1)
#Bradykinesia
df_clinical['Bradykinesia'] = df_clinical[['3.2', '3.4 D', '3.4 E', '3.5 D', 
                                           '3.5 E', '3.6 D', '3.6 E', '3.7 D', 
                                           '3.7 E', '3.8 D', '3.8 E', '3.9', 
                                           '3.14']].sum(axis=1)
#Rigidity
df_clinical['Rigidity'] = df_clinical[['3.3 neck', '3.3 MSD', '3.3 MSE', 
                                       '3.3 MID', '3.3 MIE']].sum(axis=1)
#Total UPDRS
df_clinical['Total_UPDRS_III'] = df_clinical[['3.1', '3.2', '3.3 neck', '3.3 MSD', 
                                              '3.3 MSE', '3.3 MID', '3.3 MIE', 
                                              '3.4 D', '3.4 E', '3.5 D', '3.5 E', 
                                              '3.6 D', '3.6 E', '3.7 D', '3.7 E', 
                                              '3.8 D', '3.8 E', '3.9', '3.10', 
                                              '3.11', '3.12', '3.13', '3.14', 
                                              '3. 15 D', '3.15 E', '3.16 D', 
                                              '3. 16 E', '3.17 MSD', 
                                              '3.17 MSE', '3.17 MID', '3.17 MIE', 
                                              '3.17 jaw', '3.18']].sum(axis=1)
#Rest tremor
df_clinical['Rest_tremor'] = df_clinical[['3.17 MSD', '3.17 MSE', '3.17 MID', 
                                          '3.17 MIE', '3.17 jaw']].sum(axis=1)
#Action tremor
df_clinical['Action_tremor'] = df_clinical[['3. 15 D', '3.15 E', '3.16 D', 
                                            '3. 16 E']].sum(axis=1)
#Posture
df_clinical['Posture'] = df_clinical[['3.12', '3.13']].mean(axis=1)

#Path and read imaging data
path_datscan = r'C:\Users\Admin\Desktop\Pedro\CCU\PD-DaTscan\DaTScan_Analise\Prog\DaTscan_Stat_v2.csv'
df_datscan = pd.read_csv(path_datscan, sep=';', index_col='ID')

#Compute mean striatum, caudate and putamen binding potentials
df_datscan['Striatum_BP'] = df_datscan[['Striatum BP (left)', 
                                        'Striatum BP (right)']].mean(axis=1)
df_datscan['Caudate_BP'] = df_datscan[['Caudate BP (left)', 
                                       'Caudate BP (right)']].mean(axis=1)
df_datscan['Putamen_BP'] = df_datscan[['Putamen BP (left)', 
                                       'Putamen BP (right)']].mean(axis=1)

#Prepare final dataframe
df_total = pd.concat([df_clinical, df_datscan, df_kinematics], axis=1)
df_total = df_total[df_total['group'].notna()]

#Log of max power between 4-6 Hz
df_total['log_X_ratio_welch'] = np.log(df_total['X_ratio_welch'])

#%%
"""
Correlations and bar plots. Each plot_stats command will generate two correlation 
plots and one bar plot as follows:
    1 correlation plot with stats for each group
    1 correlation plot with stats for the whole cohort
    1 bar plot for the y variable
Set x and y variables as desired, based on the columns available at df_total.
Example below is a correlation between log max power 4-6 hz and the caudate
binding potential.

Optionally, df_total can be split based on the group argument (positive or 
negative), which will generate individual group plots
"""

plt.figure()
plot_stats(data=df_total, x='log_X_ratio_welch', y='Caudate_BP', hue='group')
plt.show()

#Set group as 'positive' or 'negative' to generate individual group plots
df_total2 = df_total[df_total['group']=='Positive']

plt.figure()
#call plot_stats with df_total2 set for positive group
plot_stats(data=df_total2, x='log_X_ratio_welch', y='Caudate_BP', hue='group')
plt.show()

#%%

"""
Linear Regression models, X1 set as group of predictors and y as variable to be
predicted. df_for_pred can be changed to look at positive or negative groups
separately.
"""

import statsmodels.api as sm

#Split total dataframe into positive and negative. Used for group-wise predictions
df_pos = df_total.loc[df_total['group'] == 'Positive']
df_neg = df_total.loc[df_total['group'] == 'Negative']

#Set df_for_pred based on the desired group for prediction
df_for_pred = df_pos

#Set which variables will be used to predict Max power 4-6Hz (log)
X1 = df_for_pred[['Rest_tremor', 'Bradykinesia', 'Rigidity']] 

#Add constant to model
X_for_pred = sm.add_constant(X1)
#Set which variable will be predicted
y = df_for_pred[['log_X_ratio_welch']]

#Run model using statsmodels ordinary least squares
model = sm.OLS(y, X_for_pred, missing='drop')
#Print model results
print(model.fit().summary())


