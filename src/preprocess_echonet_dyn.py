import numpy as np
import utils
import pandas as pd
import os 
import h5py


# 
def preprocess_echonet_dynamic(path_to_csv, path_to_tracings, path_to_data, save_path):
    # create a list of file names 
    list_of_files = os.listdir(path_to_data)

    # manual list for excluding files due to missingness of tracings 
    list_to_exclude = [
        '0X5DD5283AC43CCDD1.avi',
        '0X234005774F4CB5CD.avi',
        '0X2DC68261CBCC04AE.avi',
        '0X35291BE9AB90FB89.avi',
        '0X6C435C1B417FDE8A.avi',
        '0X5515B0BD077BE68A.avi'
    ]

    list_of_diffs = [f for f in list_of_files if f not in list_to_exclude]

    print('excluded elements:', len(list_of_files) - len(list_of_diffs))
    # load the csv
    csv_file = pd.read_csv(path_to_csv)
    # iterate over all file names and load the video, load the segmentation masks, store it into an h5 file in the correct folder
    off_count = []
    for file in list_of_diffs:
    # for file in list_of_files:
        try:
            split = csv_file[csv_file['FileName'] == file[:-4]]['Split'].values[0]
        except:
            print(f'something wring with {file}')
            continue
        save_path_file = os.path.join(save_path, split, file[:-4] + '.h5')
        if os.path.exists(save_path_file):
            print(f"Skipping {file}, already processed.")
            continue
        
        # load tensor from file 
        file_path = os.path.join(path_to_data, file)
        video_tensor = utils.avi2video(file_path) 
        # create binary segmentations 
        mask_ed, mask_es, frame_ed, frame_es = utils.get_segms_from_traces_filename(file, path_to_tracings)
        # check whether it is train, val or test
        
        

        # extract ejection fraction
        efr = csv_file[csv_file['FileName'] == file[:-4]]['EF'].values[0]

        # compute efr from masks to store
        traces_mask_ed = utils.extract_lv_axes(mask_ed)
        traces_mask_es = utils.extract_lv_axes(mask_es)
        vol_ed_mask = utils.calculate_volume_from_tracings(traces_mask_ed)
        vol_es_mask = utils.calculate_volume_from_tracings(traces_mask_es)
        efr_mask = ((vol_ed_mask - vol_es_mask) / vol_ed_mask) * 100

        # save into h5 file
        
        
        # check for too big difference or implausible values
        difference = np.abs(efr_mask - efr)
        if (efr_mask < 0) or (efr < 0) or difference > 10:
            print(save_path_file)
            print('gt lvef', efr)
            print('mask lvef', efr_mask)
            off_count.append(file)

        
        
        else:
            f = h5py.File(save_path_file, 'w')
            f.create_dataset(f'ed/image', data=video_tensor[frame_ed])
            f.create_dataset(f'ed/mask', data=mask_ed)
            f.create_dataset(f'ed/frame', data=frame_ed)

            f.create_dataset(f'es/image', data=video_tensor[frame_es])
            f.create_dataset(f'es/mask', data=mask_es)
            f.create_dataset(f'es/frame', data=frame_es)

            f.create_dataset('efr', data=efr)

            f.close()
    print(len(off_count))
    print(off_count)
    return

path_to_csv = '/home/paul/Downloads/EchoNet-Dynamic/FileList.csv'
path_to_videos = '/home/paul/Downloads/EchoNet-Dynamic/Videos/'
path_to_tracings = '/home/paul/Downloads/EchoNet-Dynamic/VolumeTracings.csv'
save_path = '/home/paul/Downloads/echonet_dynamic_preprocessed/'

preprocess_echonet_dynamic(path_to_csv, path_to_tracings, path_to_videos, save_path)