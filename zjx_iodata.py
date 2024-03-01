import pandas as pd
import numpy as np
import os

path = '//FS/zeng/free-moving-track/240201-AIY-9DA/pc3'

def findfile(path):
    for root, ds, fs in os.walk(path):
        for f in fs:
            if 'iodata' in f and f.endswith('.csv'):
                file = os.path.join(root, f)
                yield file

for file in findfile(path):
    print(file)
    df = pd.read_csv(file, sep=' ', header=None)
    column_names = ['time(sec)', 'x_stage(um)', 'y_stage(um)', 'z_stage(um)', 'x_image(pixel)', 'y_image(pixel)', 'charTTL']
    df.columns = column_names
    df['48->52'] = df['charTTL'].diff()
    df2 = df.copy()
    df['dt'] = df['time(sec)'].diff()
    dxy = np.sqrt((df.iloc[:, 1].diff(). pow(2)) + (df.iloc[:, 2].diff(). pow(2)))
    df['dxy'] = dxy


    df2 = df2[df2['48->52'] == 4]
    df2['dt'] = df2['time(sec)'].diff()
    dxy2 = np.sqrt((df2.iloc[:, 1].diff().pow(2)) + (df2.iloc[:, 2].diff().pow(2)))
    df2['dxy'] = dxy2

    folder_path, file_name = os.path.split(file)
    new_file_name1 = 'new_' + file_name
    new_file_name2 = 'edited_' + file_name

    df.to_csv(os.path.join(folder_path, new_file_name1), index=False)
    df2.to_csv(os.path.join(folder_path, new_file_name2), index=False)


def findfile2(path):
    for root, ds, fs in os.walk(path):
        for f in fs:
            if 'img_time_' in f and f.endswith('.csv'):
                file = os.path.join(root, f)
                yield file

for file2 in findfile2(path):
    print(file2)
    df_2 = pd.read_csv(file2, sep=' ', header=None)
    column_names2 = ['frames', 'time(sec)']
    df_2.columns = column_names2
    df_2['dt'] = df_2['time(sec)'].diff()
    folder_path2, file_name2 = os.path.split(file2)

    new_file_name_1 = 'edited_' + file_name2

    df_2.to_csv(os.path.join(folder_path2, new_file_name_1), index=False)




