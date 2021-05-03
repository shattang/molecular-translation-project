import os
import glob
import shutil
import argparse
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    args = parser.parse_args()

    model_dir = args.model_dir or '.'
    files = glob.glob(os.path.join(model_dir, 'model_epoch_*.pth'))

    data = []
    for file_path in files:
        fname = os.path.split(file_path)[1]
        split = fname.split('_')
        data.append([int(split[2]), int(split[3]), file_path])
    df = pd.DataFrame(data, columns=['epoch', 'iter_num', 'fpath'])
    df.sort_values(['epoch','iter_num'], inplace=True)
    #print(df.to_string())
    idxs = df.groupby(['epoch']).iter_num.idxmax().values
    #print(df.loc[idxs].to_string())
    keepfiles = set(df.loc[idxs].fpath.values)
    #print(keepfiles)

    backup_dir = os.path.join(model_dir,'backup')
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    for fpath in files:
        if not fpath in keepfiles:
            print('move {} {}'.format(fpath, backup_dir))
            shutil.move(fpath, backup_dir)
