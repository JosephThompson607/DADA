
import pandas as pd
def read_and_format(csv_fp):
    df = pd.read_csv(csv_fp)
    original_sols = df[df['nodes']=='SALBP_original'].copy()
    original_sols = original_sols[['instance', 'no_stations','cpu']]
    original_sols.rename(columns={'no_stations':'s_orig', 'cpu':'orig_cpu'},inplace=True)
    df = df[df['nodes']!='SALBP_original']
    df = pd.merge(df, original_sols, on=['instance'])
    return df

def res_to_classification(res_df):
    res_df_off = res_df[res_df['no_stations'] > res_df['s_gt']]
    print(f"there are {res_df_off['instance'].nunique()} instances that do not match the 1 results")

    res_df['false_positive'] = (res_df['no_stations'] < res_df['s_orig']) & (res_df['s_gt'] >= res_df['s_orig_gt'])
    res_df[res_df['false_positive']==True]

    res_df['false_negative'] = (res_df['no_stations'] >= res_df['s_orig']) & (res_df['s_gt'] < res_df['s_orig_gt'])
    res_df[res_df['false_negative'] ==True]
    res_df['true_positive'] = (res_df['no_stations'] < res_df['s_orig']) & (res_df['s_gt'] < res_df['s_orig_gt'])
    res_df[res_df['true_positive'] ==True]
    res_df['true_negative'] = (res_df['no_stations'] >= res_df['s_orig']) & (res_df['s_gt'] >= res_df['s_orig_gt'])
    res_df[res_df['true_negative'] ==True]

    tp = sum(res_df['true_positive'] ==True)
    tn = sum(res_df['true_negative'] ==True)
    fn = sum(res_df['false_negative']==True)
    fp = sum(res_df['false_positive']==True)
    total = (res_df.shape[0])
    print("total of classes ", tp + tn + fn + fp, " total ", total, " total positives ", tp+fn)
    print(f" tp: {tp} , fp: {fp}, tn: {tn}, fn: {fn}")

    precision = tp/(tp+fp)
    recall = tp /(tp+fn)
    f1 =2 * precision*recall / (precision+recall)
    accuracy = (tp + tn) / (tp + fn + tn+ fp)
    print("precision ", precision, " recall" , recall, " f1 ", f1, "accuracy",accuracy)
    return res_df, (precision, recall, f1, accuracy)