
import numpy as np
from utilies import sk_util as sk
import pandas as pd
import glob
import matplotlib.pyplot as plt


# Create a list of all CSV files in the current directory
sk_pathFiles = "D:/Projects In Master Degree/Projects/Gait Analytics/Code/Codes/Datasets/Normal Working/"
sk_pathFiles1 = "D:/Projects In Master Degree/Projects/Gait Analytics/Code/Codes/Datasets/"
# # Get list of file names in directory
# file_list = os.listdir(sk_pathFiles)

# # Initialize empty dataframe to store data
# sk_dataLeft = pd.DataFrame()
# sk_dataRight = pd.DataFrame()
# sk_sensor = []

all_files = glob.glob(f"{sk_pathFiles}*.csv")

# Create an empty list to store dataframes
dfs = []

# Loop through all CSV files and read them into dataframes
for file in all_files:
    df = pd.read_csv(file, parse_dates=['strDate'])
    dfs.append(df)

# Concatenate all dataframes into a single dataframe
combined_df = pd.concat(dfs)

# Reset the index of the combined dataframe
combined_df = combined_df.reset_index(drop=True)

# Save the combined dataframe to a new CSV file
combined_df.to_csv(f"{sk_pathFiles1}combined_data.csv", index=False)
# sk_pathData = pd.read_csv(f"{sk_pathFiles}20230404-20227613.csv")
sk_dataFrame = pd.DataFrame(combined_df)
# sk_dataRight.columns = ['ID', 'UserID', 'strDate', 'NameDevices', 'AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'AngleX', 'AngleY', 'AngleZ', 'MagX', 'MagY', 'MagZ']
# sk_dataRight.columns = ['ID', 'UserID', 'strDate', 'NameDevices', 'AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'AngleX', 'AngleY', 'AngleZ', 'MagX', 'MagY', 'MagZ']

sk_dataLeft, sk_dataRight, sk_sensor = sk.sk_separate_data_2_lr(sk_dataFrame)

columns= ['UserID', 'strDate', 'NameDevices', 'AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'AngleX', 'AngleY', 'AngleZ', 'MagX', 'MagY', 'MagZ']
df_l = pd.DataFrame(sk_dataLeft, columns=columns)
df_l = df_l.reset_index(drop=True)
df_l = df_l.sort_values('strDate')
print(df_l)

columns= ['UserID', 'strDate', 'NameDevices', 'AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'AngleX', 'AngleY', 'AngleZ', 'MagX', 'MagY', 'MagZ']
df_R = pd.DataFrame(sk_dataRight, columns=columns)
df_R = df_R.reset_index(drop=True)
df_R = df_R.sort_values('strDate')
print(df_R)

synchronized = pd.merge_asof(df_l, df_R, on='strDate', direction='nearest')
# UserID_x	strDate	NameDevices_x	AccX_x	AccY_x	AccZ_x	GyroX_x	GyroY_x	GyroZ_x	AngleX_x	AngleY_x	AngleZ_x	MagX_x	MagY_x	MagZ_x	UserID_y	NameDevices_y	AccX_y	AccY_y	AccZ_y	GyroX_y	GyroY_y	GyroZ_y	AngleX_y	AngleY_y	AngleZ_y	MagX_y	MagY_y	MagZ_y
synchronized = synchronized.rename(columns= {'UserID_x':'UserID_L', 'NameDevices_x':'NameDevices_L', 'AccX_x':'AccX_L', 'AccY_x':'AccY_L', 'AccZ_x':'AccZ_L', 'GyroX_x':'GyroX_L', 'GyroY_x':'GyroY_L', 'GyroZ_x':'GyroZ_L', 'AngleX_x':'AngleX_L', 'AngleY_x':'AngleY_L', 'AngleZ_x':'AngleZ_L', 'MagX_x':'MagX_L', 'MagY_x':'MagY_L', 'MagZ_x':'MagZ_L','UserID_y':'UserID_R', 'NameDevices_y':'NameDevices_R', 'AccX_y':'AccX_R', 'AccY_y':'AccY_R', 'AccZ_y':'AccZ_R', 'GyroX_y':'GyroX_R', 'GyroY_y':'GyroY_R', 'GyroZ_y':'GyroZ_R', 'AngleX_y':'AngleX_R', 'AngleY_y':'AngleY_R', 'AngleZ_y':'AngleZ_R', 'MagX_y':'MagX_R', 'MagY_y':'MagY_R', 'MagZ_y':'MagZ_R'})



# Plot View Data from AccX, GyroZ, and Ang_Y
fig, ax = plt.subplots(3, 1, figsize=(10, 6))
ax[0].plot(synchronized['AccX_L'])
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Amplitude')
ax[0].set_title('AccX_L')
ax[1].plot(synchronized['AngleY_L'])
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Amplitude')
ax[1].set_title('AngleY_L')
ax[2].plot(synchronized['GyroZ_L'])
ax[2].set_xlabel('Time (s)')
ax[2].set_ylabel('Amplitude')
ax[2].set_title('GyroZ_L')
plt.tight_layout()
# plt.plot(df, '-', color='black', label='Original')
plt.show()

synchronized.to_csv(f'{sk_pathFiles1}datasynchronized.csv', index=False)
# Display the synchronized data
print(synchronized)
count = synchronized.groupby(pd.Grouper(key='strDate', axis=0, freq='S')).count()
print(count)
print('------------------------------------------------------------')
print(count['NameDevices_L'].mean())
print('------------------------------------------------------------')

# plt.plot(vertical_acc, '-', color='black', label='Original')
plt.plot(synchronized["GyroZ_L"], '-', color='blue', label='Left')
plt.plot(synchronized["GyroZ_R"], "-", color='red', label='Right')
plt.legend()
plt.title('Data Synchronize')
plt.show()

sk_sensorName_L = sk_sensor[0]
sk_sensorName_L = sk_sensorName_L.replace(":","")
sk_sensorName_R = sk_sensor[1]
sk_sensorName_R = sk_sensorName_R.replace(":","")

sk_dataFileLeft = f"{sk_pathFiles1}{sk_sensorName_L}_Foot.csv"
sk_dataFileRight = f"{sk_pathFiles1}{sk_sensorName_R}_Foot.csv"

sk_dataLeft.to_csv(sk_dataFileLeft, columns=['UserID', 'strDate', 'NameDevices', 'AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'AngleX', 'AngleY', 'AngleZ', 'MagX', 'MagY', 'MagZ'], index=False)
sk_dataRight.to_csv(sk_dataFileRight, columns= ['UserID', 'strDate', 'NameDevices', 'AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'AngleX', 'AngleY', 'AngleZ', 'MagX', 'MagY', 'MagZ'], index=False)
# print(sk_dataLeft, sk_dataRight)

# print(sk_dataFrame)


