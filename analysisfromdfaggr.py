#!/usr/bin/env python

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
import pingouin as pg
from scipy.special import logit
import statsmodels.formula.api as smf

def delta(orig, dest, df):
    # List of columns to analyze
    metrics = ['ECG_Rate_Mean', 'HRV_RMSSD', 'HRV_SDNN', 'HRV_MeanNN', 'EDA_Tonic_Mean', 'EDA_Tonic_SD', 'SCR_Peaks_Amplitude_Mean', 'Sympathetic_Percent', 'SCR_Frequency_PerMin', 'Unlim_Duration_Blk']

    # Pivot - this creates a DataFrame where columns are (Metric, Segment)
    pivoted_all = df.pivot(index='Participant', columns='Segment', values=metrics)
    #print(pivoted_all)

    # Subtract the entire 'orig' slice from the 'dest' slice
    # .xs (cross-section) allows us to select all metrics for one specific Segment
    df_dest = pivoted_all.xs(dest, axis=1, level='Segment')
    df_orig = pivoted_all.xs(orig, axis=1, level='Segment')
    
    deltas = df_dest - df_orig
    
    #print(f"Deltas from {orig} to {dest}:")
    #print(deltas)
    return deltas

def fixPB12(master_df):
    # Fix PB12 with values from B12-partie_2
    # Define the segments to be replaced
    target_segments = ['stress2_MIST', 'nf2_2D']
    # Extract the "Source" data (from PB12-partie_2)
    # We make a .copy() to avoid modifying the original dataframe accidentally
    source_data = master_df[
        (master_df['Participant'] == 'PB12-partie_2') &
        (master_df['Segment'].isin(target_segments))].copy()
    # Change the ID in the source data to match the "Target" (PB12)
    source_data['Participant'] = 'PB12'
    # Remove the "Old/Corrupt" segments from the target (PB12)
    # This prevents the 'Duplicate entries' error in your delta function
    master_df = master_df.drop(
        master_df[
            (master_df['Participant'] == 'PB12') &
            (master_df['Segment'].isin(target_segments))
        ].index
    )
    # Append the "New" data from PB12-partie_2 into the master dataframe
    master_df = pd.concat([master_df, source_data], ignore_index=True)
    ## Optional: Remove the temporary 'PB12-partie_2' rows if you no longer need them
    master_df = master_df[master_df['Participant'] != 'PB12-partie_2']
    #print("Final segments for PB12:")
    #print(master_df.query("Participant == 'PB12'")['Segment'].unique())

def produce_normalized_metrics(master_df):
    log_metrics = ['EDA_Tonic_Mean', 'EDA_Tonic_SD', 'SCR_Peaks_Amplitude_Mean']
    for metric in log_metrics:
        if metric in master_df.columns:
            # Create a dataframe of just baselines
            baselines = master_df[master_df['Segment'].str.contains('baseline')].groupby('Participant')[metric].mean()
            # Map those baselines back to the main dataframe
            master_df['Participant_Baseline_' + metric] = master_df['Participant'].map(baselines)
            # Apply ln(1+x) to normalize the distribution of EDA/SCR
            master_df[metric] = np.log1p(master_df[metric]) - np.log1p(master_df['Participant_Baseline_' + metric])

    # Get average baseline per participant for SCR_Frequency_PerMin
    scr_baselines = master_df[master_df['Segment'].str.contains('baseline')].groupby('Participant')['SCR_Frequency_PerMin'].mean()
    # Map those baselines back to the main dataframe
    master_df['SCR_Freq_Delta'] = np.sqrt(master_df['SCR_Frequency_PerMin']) - np.sqrt(master_df['Participant'].map(scr_baselines))

    sym_baselines = master_df[master_df['Segment'].str.contains('baseline')].groupby('Participant')['EDA_SympatheticN'].mean()
    # Map those baselines back to the main dataframe
    master_df['Sympathetic_Delta'] = logit(master_df['EDA_SympatheticN'].clip(0.001, 0.999)) - logit(master_df['Participant'].map(sym_baselines).clip(0.001, 0.999))

    # Get average baseline per participant for ECG
    ecg_baselines = master_df[master_df['Segment'].str.contains('baseline')].groupby('Participant')['ECG_Rate_Mean'].mean()
    # Map and subtract to get the "BPM Change"
    master_df['ECG_Delta_over_baseline'] = master_df['ECG_Rate_Mean'] - master_df['Participant'].map(ecg_baselines)

    # Get average baseline per participant for HRV (RMSSD)
    hrv_baselines = master_df[master_df['Segment'].str.contains('baseline')].groupby('Participant')['HRV_RMSSD'].mean()
    # Map and subtract to get the "ms Change"
    # Negative values = Reduced regulation (higher stress) | Positive values = Increased relaxation
    master_df['HRV_Delta_over_baseline'] = master_df['HRV_RMSSD'] - master_df['Participant'].map(hrv_baselines)

def main():
    # Unpickling (deserializing) from a file
    with open('dfmaster.pkl', 'rb') as f:
        master_df = pickle.load(f)

    fixPB12(master_df)
    produce_normalized_metrics(master_df)
    #analysis_of_deltas_with_groups(master_df)
    LMM_Condition_vrfirst_abbafirst_phase(master_df)
    #LMM_Condition_x_vrfirst_abbafirst_phase(master_df)
    return 0

def analysis_of_deltas_with_groups(master_df):
    Group1 = ['PB2', 'PB19', 'PB23', 'PB24']
    Group4 = ['PB4', 'PB13', 'PB15', 'PB17', 'PB21']
    Group2 = ['PB3', 'PB5', 'PB22', 'PB26', 'PB27']
    Group3 = ['PB7', 'PB14', 'PB16', 'PB25', 'PB12']
    # Create a mapping dictionary
    group_map = {}
    for pb in Group1: group_map[pb] = 'Group1'
    for pb in Group2: group_map[pb] = 'Group2'
    for pb in Group3: group_map[pb] = 'Group3'
    for pb in Group4: group_map[pb] = 'Group4'
    # Add the column to your master_df
    master_df['Experiment_Group'] = master_df['Participant'].map(group_map)

    group_configs = {
        'Group1': [('stress1_MIST', 'nf1_VR'), ('stress2_ABBA', 'nf2_2D')],
        'Group2': [('stress1_MIST', 'nf1_2D'), ('stress2_ABBA', 'nf2_VR')],
        'Group3': [('stress1_ABBA', 'nf1_VR'), ('stress2_MIST', 'nf2_2D')],
        'Group4': [('stress1_ABBA', 'nf1_2D'), ('stress2_MIST', 'nf2_VR')]
    }
    results_storage = {}

    for group, transitions in group_configs.items():
        print(f"\n--- Analyzing {group} ---")
        for (orig, dest) in transitions:
            # Get deltas for the whole df
            all_deltas = delta(orig, dest, master_df)
            # Filter for only the members of THIS specific group
            # This prevents NaNs from other participants from appearing
            group_members = [pb for pb, g in group_map.items() if g == group]
            group_deltas = all_deltas.loc[all_deltas.index.isin(group_members)]
            # Store and Print
            label = f"{group}_{orig}_to_{dest}"
            results_storage[label] = group_deltas
            print(f"Transition: {orig} -> {dest}")
            print(group_deltas[['EDA_Tonic_Mean', 'SCR_Frequency_PerMin', 'HRV_RMSSD']].mean())

    metrics = ['ECG_Rate_Mean', 'HRV_RMSSD', 'HRV_SDNN', 'HRV_MeanNN', 'EDA_Tonic_Mean', 'EDA_Tonic_SD', 'SCR_Peaks_Amplitude_Mean', 'Sympathetic_Percent', 'SCR_Frequency_PerMin', 'Unlim_Duration_Blk']
    for group_name, configs in group_configs.items():
        print(f"\n{'='*20} {group_name} Statistical Analysis {'='*20}")
        # 1. Extract the group members (assuming you have your Group lists defined)
        group_list = eval(group_name) 
        print('group_list: '+str(group_list))
        # 2. Get Deltas for Transition 1 (T1) and Transition 2 (T2)
        t1_all = delta(configs[0][0], configs[0][1], master_df)
        t2_all = delta(configs[1][0], configs[1][1], master_df)
        # Storage for this group's results
        group_summary = []
        for m in metrics:
        # Filter for this group only
            t1_m = t1_all.loc[t1_all.index.isin(group_list)][m]
            t2_m = t2_all.loc[t2_all.index.isin(group_list)][m]
            # Ensure we have the same participants for both sessions
            common = t1_m.index.intersection(t2_m.index)
            if len(common) >= 2:
                stats = pg.ttest(t1_m.loc[common], t2_m.loc[common], paired=True)
                # Extract results (using the underscore names from your version)
                res = {
                    'Metric': m,
                    'T': stats.at['T_test', 'T'],
                    'p_val': stats.at['T_test', 'p_val'],
                    'cohen_d': stats.at['T_test', 'cohen_d'],
                    'n_pairs': len(common)
                }
                group_summary.append(res)

        # Convert to DataFrame for a clean summary table
        summary_df = pd.DataFrame(group_summary)
        # Highlight significant results (p < 0.05)
        print(summary_df.sort_values(by='p_val'))

    # Define which segments are VR and which are 2D based on your study design
    vr_segments = ['nf1_VR', 'nf2_VR']
    two_d_segments = ['nf1_2D', 'nf2_2D']
    all_summary_rows = []
    for group_name, configs in group_configs.items():
        group_list = eval(group_name)
        for (orig, dest) in configs:
            # 1. Get the deltas for this specific transition
            current_deltas = delta(orig, dest, master_df)
            # 2. Filter for group members
            group_deltas = current_deltas.loc[current_deltas.index.isin(group_list)]
            # 3. Label the modality (VR or 2D)
            modality = 'VR' if dest in vr_segments else '2D'
            # 4. Calculate the average for each metric
            for metric in metrics:
                all_summary_rows.append({
                    'Group': group_name,
                    'Modality': modality,
                    'Metric': metric,
                    'Mean_Delta': group_deltas[metric].mean(),
                    'Std_Error': group_deltas[metric].sem(),
                    'N': len(group_deltas)
                })
    # Convert to DataFrame
    summary_df = pd.DataFrame(all_summary_rows)
    # Create a Pivot Table for the final report
    final_report = summary_df.pivot_table(
        index='Metric',
        columns='Modality',
        values='Mean_Delta',
        aggfunc='mean'
    )
    print("\n")
    print("=== FINAL COMPARISON: VR vs 2D RELAXATION (Across All Groups) ===")
    print(final_report)

    # Filter for just the most important metrics
    plot_metrics = ['EDA_Tonic_Mean', 'SCR_Frequency_PerMin', 'HRV_RMSSD']
    df_plot = summary_df[summary_df['Metric'].isin(plot_metrics)]
    sns.barplot(data=df_plot, x='Metric', y='Mean_Delta', hue='Modality')
    plt.title("Physiological Recovery: VR vs 2D")
    #plt.show()
    fig = plt.gcf()
    fig.savefig("Physiological_Recovery_VR_vs_2D.png")
    plt.close()

    # Prepare to collect all data points
    vr_pool = []
    two_d_pool = []
    for group_name, configs in group_configs.items():
        group_list = eval(group_name)
        for (orig, dest) in configs:
            # Get deltas for this specific transition
            current_deltas = delta(orig, dest, master_df)
            # Filter for the relevant group members
            group_deltas = current_deltas.loc[current_deltas.index.isin(group_list)].copy()
            # Assign a modality label
            if 'VR' in dest:
                vr_pool.append(group_deltas)
            elif '2D' in dest:
                two_d_pool.append(group_deltas)
    # Create the Pooled DataFrames
    df_vr = pd.concat(vr_pool)
    df_two_d = pd.concat(two_d_pool)
    # Calculate T-Tests for every metric
    final_results = []
    for m in metrics:
        # We use independent t-test (not paired) because we are pooling across different sessions/people
        res = pg.ttest(df_vr[m].dropna(), df_two_d[m].dropna(), paired=False)
        final_results.append({
            'Metric': m,
            'Mean_VR': df_vr[m].mean(),
            'Mean_2D': df_two_d[m].mean(),
            'p_val': res.at['T_test', 'p_val'],
            'cohen_d': res.at['T_test', 'cohen_d'],
            'Significant': 'YES' if res.at['T_test', 'p_val'] < 0.05 else 'no'
        })
    # Display the Final Summary
    final_summary_df = pd.DataFrame(final_results).sort_values('p_val')
    print("\n")
    print("=== FINAL POOLED COMPARISON: VR vs 2D MODALITY ===")
    print(final_summary_df)

    # Define the output path
    file_name = 'Study_Results_5min.xlsx'
    with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
        # Save the raw Master DataFrame
        master_df.to_excel(writer, sheet_name='Raw_Data', index=False)
        # Save the Final Pooled Comparison (VR vs 2D)
        # final_summary_df is the one we created in the last step
        final_summary_df.to_excel(writer, sheet_name='Final_Stats', index=False)
        # Save Group-specific results
        # You can loop through your groups to add them as individual sheets
        for group_name in ['Group1', 'Group2', 'Group3', 'Group4']:
            group_data = master_df[master_df['Experiment_Group'] == group_name]
            group_data.to_excel(writer, sheet_name=f'Data_{group_name}', index=False)
    print(f"Successfully exported all results to {file_name}")

    # Save the full processed dataset (Long Format)
    # This includes every participant, segment, and NeuroKit2 metric
    master_df.to_csv('Master_Data_5min.csv', index=False)
    # Save the final statistical comparison (VR vs 2D)
    # This contains your p-values and Cohen's d effect sizes
    final_summary_df.to_csv('Final_Statistical_Results_VR_vs_2D_5min.csv', index=False)
    # Save the Delta values (the actual change scores)
    # This is useful for creating plots in other software like R or SPSS
    all_deltas.to_csv('Participant_Deltas_Summary_5min.csv', index=True)
    # Save Group-specific results
    # You can loop through your groups to add them as individual sheets
    for group_name in ['Group1', 'Group2', 'Group3', 'Group4']:
        group_data = master_df[master_df['Experiment_Group'] == group_name]
        group_data.to_csv(f'Data_{group_name}_5min.csv', index=False)
    print("CSVs exported: Master_Data_5min.csv, Final_Statistical_Results_VR_vs_2D_5min.csv and Data_* for groups")
    print("\n")

def LMM_Condition_vrfirst_abbafirst_phase(master_df):
    import statsmodels.formula.api as smf
    clean_df = master_df.copy()
    # Select rows where 'baseline' is NOT in the Segment string
    clean_df = clean_df[~clean_df['Segment'].str.contains('baseline', case=False, na=False)]
    clean_df['Condition'] = clean_df['Segment'].apply(lambda x: 'VR' if 'VR' in x else '2D')
    # Try to use linearly independent conditions
    # 1. Identify participants who have at least one segment named 'nf1_VR'
    vr_first_participants = clean_df.loc[clean_df['Segment'] == 'nf1_VR', 'Participant'].unique()
    # 2. Create the column: If the participant is in that list, mark all their rows as 'VRfirst'
    clean_df['vrfirst'] = clean_df['Participant'].apply(
        lambda x: 'VRfirst' if x in vr_first_participants else 'VRlast'
    )
    # 1. Identify participants who have at least one segment named 'stress1_ABBA'
    abba_first_participants = clean_df.loc[clean_df['Segment'] == 'stress1_ABBA', 'Participant'].unique()
    # 2. Create the column: If the participant is in that list, mark all their rows as 'ABBAfirst'
    clean_df['abbafirst'] = clean_df['Participant'].apply(
        lambda x: 'ABBAfirst' if x in abba_first_participants else 'ABBAlast'
    )
    # Define the conditions
    conditions = [
        clean_df['Segment'].str.contains('nf1', case=False, na=False),
        clean_df['Segment'].str.contains('nf2', case=False, na=False)
    ]
    # Define the corresponding values
    choices = ['phase1', 'phase2']
    # Create the column (default to 'other' if neither nf1 nor nf2 is found)
    clean_df['Phase'] = np.select(conditions, choices, default='other')
    #clean_df['EDA_Tonic_Mean'] = pd.to_numeric(clean_df['EDA_Tonic_Mean'], errors='coerce')
    clean_df = clean_df.dropna(subset=['EDA_Tonic_Mean']).reset_index(drop=True)

    metrics = []
    pvalues = []
    for metric, title in [ ("SCR_Freq_Delta", "SCR Freq Delta"),
                          ("Sympathetic_Delta", "EDA Sympathetic Delta"),
                          #("EDA_Tonic_Mean", "EDA Tonic Mean"),
                          #("ECG_Delta_over_baseline", "ECG Delta over baseline"),
                          ("HRV_Delta_over_baseline", "HRV RMSSD Delta over baseline")]:
        formula = metric + " ~ Condition + vrfirst + abbafirst + Phase"
        result = LMM_runmodel(clean_df, formula, title)
        metrics.append(metric)
        pvalues.append(result.pvalues["Condition[T.VR]"])
        
    summary_df = get_fdr(metrics,pvalues)
    print("--- FDR (Benjamini-Hochberg) for main metrics ---")
    print(summary_df)
    print("\n")

    do_VIF_report(clean_df)

    # Calculate descriptive statistics for the key metrics
    # Create the summary table
    desc_table = clean_df.groupby(['vrfirst', 'Condition'])[metrics].agg(['mean', 'std']).round(3)
    # Reset index for a cleaner look if exporting to CSV
    desc_table.to_csv("Appendix_Descriptives_main_efffect_models_normalized.csv")
    print("--- Descriptive statistics report for the key metrics (normalized) ---")
    print(desc_table)
    print("\n")

    rawmetrics = ['SCR_Frequency_PerMin', 'Sympathetic_Percent', 'HRV_RMSSD']
    desc_table = clean_df.groupby(['vrfirst', 'Condition'])[rawmetrics].agg(['mean', 'std']).round(3)
    # Reset index for a cleaner look if exporting to CSV
    desc_table.to_csv("Appendix_Descriptives_main_efffect_models.csv")
    print("--- Descriptive statistics report for the key metrics (raw) ---")
    print(desc_table)
    print("\n")
    print("(Sympathetic_Percent = EDA_SympatheticN * 100)")

def calculate_fdr_for_model(result):
    from statsmodels.stats.multitest import multipletests
    # Get all p-values except the Intercept
    p_series = result.pvalues.drop('Intercept')
    # Apply Benjamini-Hochberg
    reject, p_adj, _, _ = multipletests(p_series, method='fdr_bh')
    # Create a clean results table
    fdr_df = pd.DataFrame({
        'Predictor': p_series.index,
        'Raw_P': p_series.values,
        'FDR_Adjusted_P': p_adj,
        'Significant': reject
    })
    return fdr_df

def get_fdr(metrics,pvalues):
    from statsmodels.stats.multitest import multipletests
    # Create a summary dataframe
    summary_df = pd.DataFrame({
        'Metric': metrics,
        'Raw_P': pvalues
    })
    # Apply FDR (Benjamini-Hochberg)
    # returns: (reject_array, corrected_p_array, alphacSidak, alphacBonf)
    results = multipletests(summary_df['Raw_P'], method='fdr_bh')
    summary_df['Adjusted_P'] = results[1]
    summary_df['Significant'] = results[0]
    return(summary_df)

def get_contrast(modelresult, formula):
    # 1. Get the names of the coefficients
    fe_names = modelresult.model.exog_names
    # 2. Find the index of the condition
    try:
        target_idx = fe_names.index(formula)
        # 3. Create a 2D matrix (1 row, N columns)
        # This satisfies the internal check: r_matrix.shape[1] == self.k_fe
        contrast_matrix = np.zeros((1, len(fe_names)))
        contrast_matrix[0, target_idx] = 1
        # 4. Run the test
        condition_contrast = modelresult.t_test(contrast_matrix)
        print("Contrast for " + formula)
        print(condition_contrast)
    except ValueError:
        print("Warning: 'Condition[T.VR]' not found in this model's fixed effects.")

def get_all_contrasts(result):
    fe_names = result.model.exog_names
    contrast_results = []

    for name in fe_names:
        # Create the 2D contrast matrix (1 x k_fe)
        contrast_matrix = np.zeros((1, len(fe_names)))
        target_idx = fe_names.index(name)
        contrast_matrix[0, target_idx] = 1

        # Run the test
        t_test_res = result.t_test(contrast_matrix)

        # Calculate Effect Sizes:
        z_val = t_test_res.tvalue[0][0]
        n_obs = result.nobs

        # Calculate r (correlation coefficient effect size)
        r_effect = np.abs(z_val) / np.sqrt(z_val**2 + n_obs)

        # Calculate Cohen's d equivalent
        d_effect = (2 * z_val) / np.sqrt(n_obs)

        # Extract statistics
        # result.t_test returns a ContrastResults object
        contrast_results.append({
            "Effect": name,
            "Coefficient": t_test_res.effect[0],
            "Std.Err": t_test_res.sd[0],
            "z-stat": t_test_res.tvalue[0][0], # statsmodels calls it tvalue even for z
            "p-value": t_test_res.pvalue,
            "r_effect": r_effect,
            "d_effect": d_effect
        })
    return pd.DataFrame(contrast_results)

def get_standardized_beta(clean_df, modelresult, outcome):
    # Manual calculation for Condition[T.VR]
    b = modelresult.params['Condition[T.VR]']
    sd_y = clean_df['Sympathetic_Delta'].std()
    # For a dummy variable (0 or 1), SD is sqrt(p * (1-p))
    # Or simply take the std of the encoded column
    condition_numeric = (clean_df['Condition'] == 'VR').astype(int)
    sd_x = condition_numeric.std()
    std_beta = b * (sd_x / sd_y)
    return std_beta

def LMM_runmodel(clean_df, formula, title):
    # 'Condition' is your categorical variable
    #model = smf.mixedlm("EDA_Tonic_Mean ~ Condition + Other_Predictors",
    model = smf.mixedlm(formula,
                        data=clean_df,
                        groups=clean_df["Participant"])
    result = model.fit()
    print("=== Statsmodels Linear Mixed Model " + formula + " ===")
    print(result.summary())
    fdr_table = calculate_fdr_for_model(result)
    print("--- FDR (Benjamini-Hochberg) for model " + formula + " ---")
    print(fdr_table)
    print("\n")
    m_r2, c_r2 = calculate_r2_mixed(result)
    print("--- R2 Report "+title+" ---")
    print(f"Marginal R2 (Fixed effects): {m_r2:.3f}")
    print(f"Conditional R2 (Total model): {c_r2:.3f}")
    print("\n")
    print("--- Estimated Marginal Means ("+title+") ---")
    if "Condition * vrfirst" in formula:
        emms = calculate_emmeans(result) # Use your result object
        for key, value in emms.items():
            print(f"{key}: {value:.4f}")
    else:
        emms_table = calculate_emms_main_effects(result, clean_df)
        print(emms_table.to_string(index=False))
    print("\n")

    df_contrasts = get_all_contrasts(result)
    print("--- Contrasts and Effect Sizes: (" + title + ") ---")
    print(df_contrasts)
    print("\n")
    outcome = formula.split()[0]
    std_beta = get_standardized_beta(clean_df, result, outcome)
    print(f"Standardized Beta for {outcome}: {std_beta:.3f}")
    print("\n")
    ci_table = result.conf_int()
    ci_table.columns = ['Lower 95%', 'Upper 95%']
    ci_table['Coef'] = result.params
    # Let's look at just the Fixed Effects (ignoring the Group Var)
    print("--- Confidence Intervals: (" + title + ") ---")
    print(ci_table.iloc[:-1, :])
    print("\n")
    check_model_health(result, title)
    # Export coefficients to CSV
    summary_df = result.summary().tables[1]
    title_wo_blanks = "_".join(title.split())
    summary_df.to_csv('LMM_Statsmodels_'+title_wo_blanks+'.csv')

    return result

def do_VIF_report(clean_df):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant
    # Select your fixed effects
    features = clean_df[['Condition', 'vrfirst', 'abbafirst', 'Phase']]
    # Convert to dummies (0 and 1)
    X = pd.get_dummies(features, drop_first=True).astype(float)
    X = add_constant(X)
    # Calculate VIF for each variable
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    print("--- VIF Report ---")
    print(vif_data)
    print("\n")

def LMM_Condition_x_vrfirst_abbafirst_phase(master_df):
    import statsmodels.formula.api as smf
    clean_df = master_df.copy()
    # Select rows where 'baseline' is NOT in the Segment string
    clean_df = clean_df[~clean_df['Segment'].str.contains('baseline', case=False, na=False)]
    clean_df['Condition'] = clean_df['Segment'].apply(lambda x: 'VR' if 'VR' in x else '2D')
    # Try to use linearly independent conditions
    # 1. Identify participants who have at least one segment named 'nf1_VR'
    vr_first_participants = clean_df.loc[clean_df['Segment'] == 'nf1_VR', 'Participant'].unique()
    # 2. Create the column: If the participant is in that list, mark all their rows as 'VRfirst'
    clean_df['vrfirst'] = clean_df['Participant'].apply(
        lambda x: 'VRfirst' if x in vr_first_participants else 'VRlast'
    )
    # 1. Identify participants who have at least one segment named 'stress1_ABBA'
    abba_first_participants = clean_df.loc[clean_df['Segment'] == 'stress1_ABBA', 'Participant'].unique()
    # 2. Create the column: If the participant is in that list, mark all their rows as 'ABBAfirst'
    clean_df['abbafirst'] = clean_df['Participant'].apply(
        lambda x: 'ABBAfirst' if x in abba_first_participants else 'ABBAlast'
    )
    # Define the conditions
    conditions = [
        clean_df['Segment'].str.contains('nf1', case=False, na=False),
        clean_df['Segment'].str.contains('nf2', case=False, na=False)
    ]
    # Define the corresponding values
    choices = ['phase1', 'phase2']
    # Create the column (default to 'other' if neither nf1 nor nf2 is found)
    clean_df['Phase'] = np.select(conditions, choices, default='other')
    #clean_df['EDA_Tonic_Mean'] = pd.to_numeric(clean_df['EDA_Tonic_Mean'], errors='coerce')
    clean_df = clean_df.dropna(subset=['EDA_Tonic_Mean']).reset_index(drop=True)
    LMM_runmodel(clean_df, "EDA_Tonic_Mean ~ Condition * vrfirst + abbafirst + Phase", "EDA Tonic Mean")
    plot_normalized_interaction(clean_df)
    LMM_runmodel(clean_df, "SCR_Freq_Delta ~ Condition * vrfirst + abbafirst + Phase", "SCR Freq Delta")
    LMM_runmodel(clean_df, "Sympathetic_Delta ~ Condition * vrfirst + abbafirst + Phase", "EDA Sympathetic Delta")
    LMM_runmodel(clean_df, "ECG_Delta_over_baseline ~ Condition * vrfirst + abbafirst + Phase", "ECG Delta over baseline")
    plot_dual_interaction(clean_df[clean_df['Segment'].str.contains('baseline') == False])
    LMM_runmodel(clean_df, "HRV_Delta_over_baseline ~ Condition * vrfirst + abbafirst + Phase", "HRV RMSSD Delta over baseline")
    plot_triple_signature(clean_df[~clean_df['Segment'].str.contains('baseline')])
    do_VIF_report(clean_df)
    # Calculate descriptive statistics for the key metrics
    metrics = ['EDA_Tonic_Mean', 'Sympathetic_Percent', 'SCR_Frequency_PerMin', 'ECG_Rate_Mean', 'HRV_RMSSD']
    # Create the summary table
    desc_table = clean_df.groupby(['vrfirst', 'Condition'])[metrics].agg(['mean', 'std']).round(3)
    # Reset index for a cleaner look if exporting to CSV
    # desc_table.to_csv("Appendix_Descriptives.csv")
    print("--- Descriptive statistics report for the key metrics ---")
    print(desc_table)

def calculate_emms_main_effects(result, df):
    # Extract parameters and covariance matrix
    params = result.params
    cov = result.cov_params()
    # --- 2D Condition (The Intercept) ---
    mean_2d = params['Intercept']
    se_2d = result.bse['Intercept']
    # --- VR Condition (Intercept + VR Coefficient) ---
    mean_vr = params['Intercept'] + params['Condition[T.VR]']
    # Formula for variance of a sum: Var(A+B) = Var(A) + Var(B) + 2*Cov(A,B)
    var_vr = cov.loc['Intercept', 'Intercept'] + \
             cov.loc['Condition[T.VR]', 'Condition[T.VR]'] + \
             2 * cov.loc['Intercept', 'Condition[T.VR]']
    se_vr = np.sqrt(var_vr)
    # Build the DataFrame
    emm_df = pd.DataFrame({
        'Condition': ['2D (Control)', 'VR (Treatment)'],
        'EMM (Mean)': [mean_2d, mean_vr],
        'SE': [se_2d, se_vr]
    })
    # Add 95% Confidence Intervals
    emm_df['Lower 95% CI'] = emm_df['EMM (Mean)'] - (1.96 * emm_df['SE'])
    emm_df['Upper 95% CI'] = emm_df['EMM (Mean)'] + (1.96 * emm_df['SE'])
    return emm_df

def calculate_emmeans(result):
    p = result.fe_params

    # Calculate the average 'Phase effect'
    # (Phase1_effect + Phase2_effect) / 2
    avg_phase_effect = (p['Phase[T.phase1]'] + p['Phase[T.phase2]']) / 2

    # Add this average effect to the Intercept
    base = p['Intercept'] + avg_phase_effect

    emms = {
        "VR-First + 2D": base,
        "VR-First + VR": base + p['Condition[T.VR]'],
        "VR-Last + 2D":  base + p['vrfirst[T.VRlast]'],
        "VR-Last + VR":  base + p['Condition[T.VR]'] +
                         p['vrfirst[T.VRlast]'] + p['Condition[T.VR]:vrfirst[T.VRlast]']
    }
    return emms

def plot_normalized_interaction(df):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 7))

    # 1. Create the pointplot
    ax = sns.pointplot(
        data=df,
        x='Condition',
        y='EDA_Tonic_Mean',
        hue='vrfirst',
        dodge=0.1,
        markers=["o", "s"],
        linestyles=["-", "--"],
        capsize=.1,
        errorbar=('ci', 95)
    )

    # 2. Add baseline indicator
    plt.axhline(0, color='black', linestyle=':', alpha=0.5, label='Baseline')

    # 3. DYNAMIC COLOR MATCHING
    # Get the colors Seaborn used for the lines
    # ax.get_lines() contains the error bars and the main lines
    # Usually, the first two 'collections' or lines represent our groups
    colors = [line.get_color() for line in ax.lines[::len(ax.lines)//2]]
    # Or more reliably from the legend handles:
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    palette_colors = sns.color_palette("muted", len(legend_labels))

    # 4. Add labels with matching colors
    # Finding the Y-positions for the labels based on the data means
    # vrfirst group (usually first) and vrlast group (usually second)
    # Extract the mean values to find the perfect 'anchor' for our text
    means = df.groupby(['Condition', 'vrfirst'])['EDA_Tonic_Mean'].mean()

    # Calculate Y-positions (adding a small 'buffer' of 0.05 to lift the text)
    # VR is at X-index 1
    y_pos_vrlast = means.loc[('VR', 'VRlast')] + 0.05
    y_pos_vrfirst = means.loc[('VR', 'VRfirst')] + 0.05

    # Updated Text calls
    plt.text(1.1, y_pos_vrlast, "≈ +52% jump",
         color=palette_colors[1], fontweight='bold', va='bottom', ha='left')

    plt.text(1.1, y_pos_vrfirst, "≈ -10% dip",
         color=palette_colors[0], fontweight='bold', va='bottom', ha='left')

    #plt.text(1.1, 0.45, "≈ +52% jump", color=palette_colors[1], fontweight='bold', va='center')
    #plt.text(1.1, -0.15, "≈ -10% dip", color=palette_colors[0], fontweight='bold', va='center')

    # Increase the top margin of the plot so labels don't hit the ceiling
    plt.ylim(top=plt.ylim()[1] + 0.2)

    # 5. Final touches
    plt.title('Normalized Physiological Response (Log-Ratio from Baseline)', fontsize=14, pad=20)
    plt.xlabel('Environment Condition', fontsize=12)
    plt.ylabel('Δ ln(EDA + 1) vs Baseline', fontsize=12)
    plt.legend(title='Order Group', frameon=True, loc='upper left')

    plt.tight_layout()
    plt.savefig('Normalized_EDA_Interaction_Corrected.png', dpi=300)
    #plt.show()

def calculate_r2_mixed(result):
    # 1. Fixed Effects Variance: X * beta
    fixed_fitted = np.dot(result.model.exog, result.fe_params)
    var_fixed = np.var(fixed_fitted)

    # 2. Random Effects Variance
    # result.cov_re is the covariance matrix for random effects.
    # For a random intercept model, it's a 1x1 matrix.
    # We use .iloc[0,0] if it's a DataFrame or just index it if it's an array.
    try:
        # Try to get the variance from the diagonal of the RE covariance matrix
        var_random = np.diag(result.cov_re).sum()
    except:
        # Fallback to vcomp if available
        var_random = result.vcomp[0] if len(result.vcomp) > 0 else 0

    # 3. Residual Variance (Scale)
    var_resid = result.scale

    # 4. Total Variance
    var_total = var_fixed + var_random + var_resid

    # 5. R-Squared Calculations
    marginal_r2 = var_fixed / var_total
    conditional_r2 = (var_fixed + var_random) / var_total

    return marginal_r2, conditional_r2

def plot_dual_interaction(df):
    # 1. Setup figure with two subplots
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    palette = sns.color_palette("muted", n_colors=2)

    # --- LEFT PLOT: EDA (The Significant Spike) ---
    sns.pointplot(
        data=df, x='Condition', y='EDA_Tonic_Mean', hue='vrfirst',
        ax=ax1, dodge=0.1, markers=["o", "s"], linestyles=["-", "--"],
        capsize=.1, errorbar=('ci', 95), palette=palette
    )
    ax1.axhline(0, color='black', linestyle=':', alpha=0.5)
    ax1.set_title('A: Emotional Arousal (Tonic EDA)\nSignificant Interaction (p = .016)',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel('Δ ln(EDA + 1) vs Baseline', fontsize=12)
    ax1.set_xlabel('Environment', fontsize=12)

    # Calculate positions for annotations (EDA)
    eda_means = df.groupby(['Condition', 'vrfirst'])['EDA_Tonic_Mean'].mean()
    ax1.text(1.1, eda_means.loc[('VR', 'VRlast')] + 0.02, "≈ +52% jump",
             color=palette[1], fontweight='bold', va='bottom')
    ax1.text(1.1, eda_means.loc[('VR', 'VRfirst')] - 0.05, "≈ -10% dip",
             color=palette[0], fontweight='bold', va='top')

    # --- RIGHT PLOT: ECG (The Null Result) ---
    sns.pointplot(
        data=df, x='Condition', y='ECG_Delta_over_baseline', hue='vrfirst',
        ax=ax2, dodge=0.1, markers=["o", "s"], linestyles=["-", "--"],
        capsize=.1, errorbar=('ci', 95), palette=palette
    )
    ax2.axhline(0, color='black', linestyle=':', alpha=0.5)
    ax2.set_title('B: Physical Effort (Heart Rate)\nNon-Significant Interaction (p = .095)',
                  fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylabel('Δ Heart Rate (BPM) vs Baseline', fontsize=12)
    ax2.set_xlabel('Environment', fontsize=12)

    # Clean up legends
    ax1.legend(title='Order Group', loc='upper left')
    ax2.get_legend().remove() # Only need one legend for the whole figure

    plt.tight_layout()
    plt.savefig('Physiological_Fractionation_Plot.png', dpi=300)
    #plt.show()

def plot_triple_signature(df):
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
    palette = sns.color_palette("muted", n_colors=2)

    # Common settings for all plots
    plot_params = {
        'x': 'Condition', 'hue': 'vrfirst', 'dodge': 0.1,
        'markers': ["o", "s"], 'linestyles': ["-", "--"],
        'capsize': .1, 'errorbar': ('ci', 95), 'palette': palette
    }

    # Plot 1: EDA (Sympathetic)
    sns.pointplot(data=df, y='EDA_Tonic_Mean', ax=ax1, **plot_params)
    ax1.set_title('A: Sympathetic Arousal\n(Tonic EDA)', fontweight='bold')
    ax1.set_ylabel('Δ ln(EDA + 1) vs Baseline')

    # Plot 2: Heart Rate (Demand)
    sns.pointplot(data=df, y='ECG_Delta_over_baseline', ax=ax2, **plot_params)
    ax2.set_title('B: Overall Heart Rate\n(BPM Change)', fontweight='bold')
    ax2.set_ylabel('Δ BPM vs Baseline')

    # Plot 3: HRV (Parasympathetic/Stress)
    sns.pointplot(data=df, y='HRV_Delta_over_baseline', ax=ax3, **plot_params)
    ax3.set_title('C: Vagal Tone / Regulation\n(RMSSD Change)', fontweight='bold')
    ax3.set_ylabel('Δ RMSSD (ms) vs Baseline')

    # Clean up
    for ax in [ax1, ax2, ax3]:
        ax.axhline(0, color='black', linestyle=':', alpha=0.5)
        ax.set_xlabel('Environment')

    ax1.legend(title='Order Group', loc='upper left')
    ax2.get_legend().remove()
    ax3.get_legend().remove()

    plt.tight_layout()
    plt.savefig('Physiological_Signature_Triple.png', dpi=300)
    #plt.show()

def check_model_health(result, title):
    # 'result' must be the object returned by model.fit()
    import statsmodels.api as sm
    from scipy import stats

    try:
        residuals = result.resid  # No parentheses here for statsmodels MixedLM results
        fitted_values = result.fittedvalues
    except AttributeError:
        # Fallback if the object passed was the model instead of the results
        print("Error: Please pass the FITTED result object (e.g., model.fit())")
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    # QQ-Plot: The DHARMa equivalent for checking normality
    sm.qqplot(residuals, dist=stats.norm, line='s', ax=ax[0])
    ax[0].set_title("QQ-Plot: Residual Normality: "+title)
    # Residuals vs Fitted: Checking for constant variance (Homoscedasticity)
    ax[1].scatter(fitted_values, residuals, alpha=0.5, color='teal')
    ax[1].axhline(y=0, color='red', linestyle='--')
    ax[1].set_xlabel("Fitted Values")
    ax[1].set_ylabel("Residuals")
    ax[1].set_title("Residuals vs Fitted")
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig("QQ-Plot: Residual Normality: "+title+".png")
    #plt.show()
    plt.close()

if __name__ == "__main__":
    main()

