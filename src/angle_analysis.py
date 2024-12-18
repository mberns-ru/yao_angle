import numpy as np
import scipy
import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib
import os
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.use('Agg')

# mixed model ANOVA

def run_analysis(data_folder):

    output_folder = os.path.dirname((Path(data_folder)))
    output_folder = Path(output_folder) / 'angle_results'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    folders = glob.glob(data_folder + '/*')
    
    nh_corrs = {}
    nihl_corrs = {}
    nh_incorrs = {}
    nihl_incorrs = {}

    for folder in folders:
        # Folder name format: 'animal_id hearing_status'

        folder_parsed = folder.split('_')
        hearing_status = folder_parsed[-1]
        animal_id = folder_parsed[-2][-3:]

        corrs, incorrs = extract_data(folder)

        corr_times = {'turn dur': corrs}
        corr_times = pd.DataFrame(corr_times)
        incorr_times = {'turn dur': incorrs}
        incorr_times = pd.DataFrame(incorr_times)

        if hearing_status == 'Norm':
            nh_corrs[animal_id] = corrs
            nh_incorrs[animal_id] = incorrs
            file_name = animal_id + "_nh_"
            corr_times.to_csv(str(output_folder) + '\\' + file_name + "corr.csv")
            incorr_times.to_csv(str(output_folder) + '\\' + file_name + "incorr.csv")
        else:
            nihl_corrs[animal_id] = corrs
            nihl_incorrs[animal_id] = incorrs
            file_name = animal_id + "_nihl_"
            corr_times.to_csv(str(output_folder) + '\\' + file_name + "corr.csv")
            incorr_times.to_csv(str(output_folder) + '\\' + file_name + "incorr.csv")

    # print(nh_corrs)

    for id in nh_corrs.keys():

        if id in nihl_corrs.keys():
            graph_data(id, nh_corrs[id], nihl_corrs[id], nh_incorrs[id], nihl_incorrs[id], output_folder)

        else:
            print('No matching data for ' + id)

    run_indiv_anova(nh_corrs, nihl_corrs, nh_incorrs, nihl_incorrs, output_folder)

    return

def extract_data(folder):
    # extract angle speed from folders within the folder
    angle_files = glob.glob(folder + '\\*\\*anglespeed.csv')

    corrs = []
    incorrs = []

    for file in angle_files:
        angle_speed_df = pd.read_csv(file)
        corr_rows = angle_speed_df[angle_speed_df['Score'] == 'Correct']
        incorr_rows = angle_speed_df[angle_speed_df['Score'] == 'Incorrect']

        corrs.extend(corr_rows['Decision Time (secs)'].values * 1000)
        incorrs.extend(incorr_rows['Decision Time (secs)'].values * 1000)

    return corrs, incorrs

def graph_data(id, nh_corrs, nihl_corrs, nh_incorrs, nihl_incorrs, output_folder):

    plt.bar(['Correct Pre-', 'Correct Post-', 'Incorrect Pre-', 'Incorrect Post-'], [np.mean(nh_corrs), np.mean(nihl_corrs), np.mean(nh_incorrs), np.mean(nihl_incorrs)],
            yerr=[scipy.stats.sem(nh_corrs), scipy.stats.sem(nihl_corrs), scipy.stats.sem(nh_incorrs), scipy.stats.sem(nihl_incorrs)],
            color=['black', 'orange', 'black', 'orange'])
    plt.title('Turn Duration - ' + id)
    plt.ylabel('Time (ms)')
    plt.ylim(0, 1500)
    plt.savefig(output_folder / (id + '_turn_comp.pdf'))
    plt.close()

    return

def run_anova(nh_corrs, nihl_corrs, nh_incorrs, nihl_incorrs, output_folder):

    # Format data into a dict
    data_list = []

    for id in nh_corrs.keys():
        if id in nihl_corrs.keys():
            temp_df = pd.DataFrame({
                'Subject': [id] * 4,
                'Hearing Status': ['NH', 'NIHL', 'NH', 'NIHL'],
                'Score': ['Correct', 'Correct', 'Incorrect', 'Incorrect'],
                'Average Turn Duration': [
                    np.mean(nh_corrs[id]), 
                    np.mean(nihl_corrs[id]), 
                    np.mean(nh_incorrs[id]), 
                    np.mean(nihl_incorrs[id])
                ]
            })
            data_list.append(temp_df)

    # Concatenate all DataFrames in the list into a single DataFrame
    data = pd.concat(data_list, ignore_index=True)
            
    # Run a mixed model 2-way ANOVA
    model = smf.mixedlm("Q('Average Turn Duration') ~ Q('Hearing Status') * Q('Score')", data, groups=data["Subject"])
    result = model.fit()
    #print(result.summary())

    # Save the summary to a text file
    with open(output_folder / 'anova_summary.txt', 'w') as f:
        f.write(result.summary().as_text())
    
    return

def run_indiv_anova(nh_corrs, nihl_corrs, nh_incorrs, nihl_incorrs, output_folder):

    with open(output_folder/'anova_results.txt', 'w') as file:
        # Iterate over subjects
        for subject_id in nh_corrs.keys():

            #print(subject_id)
            # Combine data for the subject
            data = {
                'turn_duration': nh_corrs[subject_id] + nihl_corrs[subject_id] + nh_incorrs[subject_id] + nihl_incorrs[subject_id],
                'hearing_status': ['nh'] * len(nh_corrs[subject_id]) + ['nihl'] * len(nihl_corrs[subject_id]) + 
                                ['nh'] * len(nh_incorrs[subject_id]) + ['nihl'] * len(nihl_incorrs[subject_id]),
                'score': ['corr'] * (len(nh_corrs[subject_id]) + len(nihl_corrs[subject_id])) + 
                        ['incorr'] * (len(nh_incorrs[subject_id]) + len(nihl_incorrs[subject_id]))
            }

            # Create a DataFrame
            df = pd.DataFrame(data)
            
            # Specify the model
            model = ols('turn_duration ~ C(hearing_status) * C(score)', data=df).fit()

            # Perform the ANOVA
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            # Write the results to the file
            file.write(f"ANOVA results for subject {subject_id}:\n")
            file.write(anova_table.to_string())
            file.write("\n" + "="*40 + "\n\n")

    return