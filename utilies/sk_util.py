import numpy as np
import matplotlib.pyplot as plt

def sk_separate_data_2_lr(dataset):
    # dataset = pd.read_csv(dataset_path, delim_whitespace= True)

    device_names = dataset["NameDevices"]
    bt_names = []
    # print("device_names: ", device_names[0])
    for i in range(len(device_names)):
        if i == 0:
            bt_names.append(device_names[0])
        elif bt_names[0] != device_names[i]:
            bt_names.append(device_names[i])
            break

    print(bt_names)  # correct

    r_data_index = np.where(dataset["NameDevices"] == bt_names[0])
    l_data_index = np.where(dataset["NameDevices"] == bt_names[1])

    r_data_df = dataset.iloc[np.asarray(r_data_index)[0], :]
    l_data_df = dataset.iloc[np.asarray(l_data_index)[0], :]

    return r_data_df, l_data_df, bt_names

def dataset_missingvalue_imputation(dataset, imputation_method, interpolation_method="cubic"):
    """
    This function performs missing value imputation based on given imputation method.
    """
    # if the imputation_method is set to interpolation, the missing values will be interpolated based on given interpolation_method
    if imputation_method == "interpolation":
        for df in dataset:
            for col in df:
                df[col].interpolate(method=interpolation_method, inplace=True)
                # print("DF:",df)

def _dataset_outliers_removing(data, threshold=1):
    """
    This function is a sub-function of dataset_outliers_removing.
    It removes any value above the given threshold, and converts it to 0.
    """
    data = data.flatten()
    nOutliers = 0
    outliers = []
    for di, dval in enumerate(data):
        if dval>threshold:
            data[di] = 0
            nOutliers += 1
            outliers.append(f"i: {di}, val: {dval}")
    return data, outliers

def dataset_outliers_removing(dataset, cols, threshold=1):
    """
    This function removes the outliers for all given columns in the dataset based on given threshold.
    """
    import numpy as np

    for dfi, dfval in enumerate(dataset):
        if dfval in cols:
            sigOut, outliers = _dataset_outliers_removing(
                data=dataset[dfval].values,
                threshold=threshold
            )
            print(f"Outliers solved for Column: {dfval} in DF_No.: {dfi}, No. of Outliers: {len(outliers)}, in {outliers}")
            dataset[dfval] = np.array(sigOut).reshape(-1, 1)


def dataset_signal_filtering(dataset, cols=[]):
    """
    This function filters the signal via Gaussian method
    """
    from scipy.ndimage.filters import gaussian_filter

    # Load data into pandas dataframe
    df = dataset
    # Select the column to smooth and filter
    
    for dfi, dfval in enumerate(dataset):
        if dfval in cols:
            # Apply gaussian_filter1d function to the selected column
            column_to_filter = dfval+"_Filtered"
            df[column_to_filter] = gaussian_filter(df[dfval], sigma=3)
    return df

# Find Combine Sensor as the root of squared maximum of acceleromater, Gyroscope, and Angle in all 1 directions of each sensor
def root_of_squared_maximum(dataset):
    _freeAcc_E = dataset['AccX_L_Filtered'].values
    _freeAcc_N = dataset['GyroZ_L_Filtered'].values
    _freeAcc_U = dataset['AngleY_L_Filtered'].values
    dataset["CSF_L"] = (_freeAcc_E**2 + _freeAcc_N**2 + _freeAcc_U**2)**0.5
    return dataset

# def dataset_signal_bandpass_filtering(dataset, cols_[]):
#     # applying a bandpass filter from data of  Optical Recordings Using a Smartphone
#     from scipy.signal import butter, filtfilt
#     # Define the filter parameters
#     lowcut = 5  # lower cutoff frequency in Hz
#     highcut = 40  # upper cutoff frequency in Hz
#     fs = 100  # sampling frequency in Hz
#     order = 4  # filter order

#     # Compute the filter coefficients using a Butterworth filter
#     nyquist = 0.5 * fs
#     low = lowcut / nyquist
#     high = highcut / nyquist
#     b, a = butter(order, [low, high], btype='band')

#     # Apply the filter to the data
#     filtered_data = filtfilt(b, a, dataset[cols_])

#     # data_ = pd.DataFrame(filtered_data)
#     return filtered_data



from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import pickle
import joblib

def sk_print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n{confusion_matrix(y_train, pred)}")
        # save the model to a .pkl file
        # with open(f'Datasets/{clf}model.pkl', 'wb') as f:
        #     pickle.dump(clf, f)
        # Save the trained classifier to a file
        joblib.dump(clf, f'Datasets/{clf}model_K.joblib')
        return pred
    
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}")
        # save the model to a .pkl file
        # with open(f'Datasets/{clf}model.pkl', 'wb') as f:
        #     pickle.dump(clf, f)
        # Save the trained classifier to a file
        joblib.dump(clf, f'Datasets/{clf}model_K.joblib')
        return pred

# def sk_print_score_predict_real(clf, real_data, y_test):
    
#     clf_report = pd.DataFrame(classification_report(y_test, real_data, output_dict=True))
#     print("Test Result:================================================")        
#     print(f"Accuracy Score: {accuracy_score(y_test, real_data) * 100:.2f}%")
#     print("_______________________________________________")
#     print(f"CLASSIFICATION REPORT:\n{clf_report}")
#     print("_______________________________________________")
#     print(f"Confusion Matrix: \n {confusion_matrix(y_test, real_data)}\n")
