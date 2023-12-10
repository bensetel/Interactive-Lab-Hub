import pandas as pd
import glob

def make_dict():
    fn = glob.glob('*.csv')
    df_dict={}
    for name in fn:
        df = pd.read_csv(name)
        df_dict.update({re.sub('.csv', '', name):df})
    return df_dict

def main():
    df_dict = make_dict()
    df_list = []
    for f, df in list(df_dict.items()):
        fn = f + '.dat'
        
        xmax = df['x'].max()
        xmin = df['x'].min()
        xmean = df['x'].mean()
        xmedian = df['x'].median()
        ymax = df['y'].max()
        ymin = df['y'].min()
        ymean = df['y'].mean()
        ymedian = df['y'].median()
        zmax = df['z'].max()
        zmin = df['z'].min()
        zmean = df['z'].mean()
        zmedian = df['z'].median()

        md = {'names':['x', 'y', 'z'], 'max':[xmax, ymax, zmax], 'min':[xmin, ymin, zmin], 'mean':[xmean, ymean, zmean], 'median':[xmedian, ymedian, zmedian]}
        ndf = pd.DataFrame.from_dict(md)
        ndf.to_csv(fn, index=False)
        

        
