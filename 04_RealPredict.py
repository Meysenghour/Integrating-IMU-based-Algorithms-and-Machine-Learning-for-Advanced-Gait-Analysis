import pickle
import pandas as pd
from utilies import sk_util as sk
import matplotlib.pyplot as plt

sk_pathFiles1 = "D:/Projects In Master Degree/Projects/Gait Analytics/Code/Codes/Datasets/"

# Test with real-time data
with open('Datasets/RandomForestClassifier(random_state=42)model.pkl', 'rb') as f:
    rf = pickle.load(f)

df_ = pd.read_csv('Datasets/Normal Working/Gait_Analysis_20227613_20230407153849287.csv', parse_dates=['strDate'])
sk_dataFrame = pd.DataFrame(df_)
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

sk_sensorName_L = sk_sensor[0]
sk_sensorName_L = sk_sensorName_L.replace(":","")
sk_sensorName_R = sk_sensor[1]
sk_sensorName_R = sk_sensorName_R.replace(":","")

sk_dataFileLeft = f"{sk_pathFiles1}{sk_sensorName_L}_Foot_real.csv"
sk_dataFileRight = f"{sk_pathFiles1}{sk_sensorName_R}_Foot_real.csv"

sk_dataLeft.to_csv(sk_dataFileLeft, columns=['UserID', 'strDate', 'NameDevices', 'AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'AngleX', 'AngleY', 'AngleZ', 'MagX', 'MagY', 'MagZ'], index=False)
sk_dataRight.to_csv(sk_dataFileRight, columns= ['UserID', 'strDate', 'NameDevices', 'AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'AngleX', 'AngleY', 'AngleZ', 'MagX', 'MagY', 'MagZ'], index=False)
print(sk_dataLeft, sk_dataRight)

df_ = pd.read_csv(sk_dataFileRight)
new_data = pd.DataFrame(df_)
## Smooth and Filter the necessary featrues by gaussian_filter1d
_col_list = ['AccX_L', 'AccY_L', 'AccZ_L','GyroX_L', 'GyroY_L', 'GyroZ_L', 'AngleX_L']
data = sk.dataset_signal_filtering(
        dataset=synchronized,
        cols=_col_list)
data.to_csv("Datasets/datafiltered.csv", columns=["GyroX_L_Filtered", "GyroY_L_Filtered", "GyroZ_L_Filtered"], index=False)
df_n = pd.read_csv('Datasets/datafiltered.csv')
real_data = rf.predict(df_n)

plt.figure(figsize=(10, 8))
plt.plot(synchronized["GyroZ_L"], '-', color='blue', label='Real Data')
plt.plot(real_data, "-", color='red', label='Data Predict')
plt.legend()
plt.title('Predict With Real Data')
plt.show()
# sk.sk_print_score_predict_real(rf, real_data, y_test)