import numpy as np
import utils
import pandas as pd
import os 
import h5py


# 
def preprocess_echonet_ped(path_to_csv, path_to_tracings, path_to_data, save_path):
    # create a list of file names 
    list_of_files = os.listdir(path_to_data)



    # load the csv
    csv_file = pd.read_csv(path_to_csv)
    # iterate over all file names and load the video, load the segmentation masks, store it into an h5 file in the correct folder
    off_count = []
    for file in list_of_files:
    # for file in list_of_files:
        try:
            split = csv_file[csv_file['FileName'] == file]['Split'].values[0]
        except:
            print(f'something wrong with {file}')
            continue
        
        # check for split
        if split < 8:
            split_final = 'TRAIN'
        elif split == 8:
            split_final = 'VAL'
        else:
            split_final = 'TEST'

        save_path_file = os.path.join(save_path, split_final, file[:-4] + '.h5')
        if os.path.exists(save_path_file):
            print(f"Skipping {file}, already processed.")
            continue
        
        # load tensor from file 
        file_path = os.path.join(path_to_data, file)
        video_tensor = utils.avi2video(file_path) 
        # create binary segmentations 
        mask_ed, mask_es, frame_ed, frame_es = utils.get_segms_from_traces_filename_ped(file, path_to_tracings)
        
        # type conversion
        try:
            frame_ed = int(frame_ed)
            frame_es = int(frame_es)
        except:
            print('no tracings for this file:', file)
            continue
        
        

        # extract ejection fraction
        efr = csv_file[csv_file['FileName'] == file]['EF'].values[0]

        # compute efr from masks to store
        try:
            traces_mask_ed = utils.extract_lv_axes(mask_ed)
            traces_mask_es = utils.extract_lv_axes(mask_es)
        except:
            print('no traces for', file)
            off_count.append(file)
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
            # print('would save')
            f = h5py.File(save_path_file, 'w')
            f.create_dataset(f'ed/image', data=video_tensor[frame_ed-1])
            f.create_dataset(f'ed/mask', data=mask_ed)
            f.create_dataset(f'ed/frame', data=frame_ed)

            f.create_dataset(f'es/image', data=video_tensor[frame_es-1])
            f.create_dataset(f'es/mask', data=mask_es)
            f.create_dataset(f'es/frame', data=frame_es)

            f.create_dataset('efr', data=efr)

            f.close()
    print(len(off_count))
    print(off_count)
    return

path_to_csv = '/home/paul/Downloads/econet_ped/echonetpediatric/pediatric_echo_avi/pediatric_echo_avi/A4C/FileList.csv'
path_to_videos = '/home/paul/Downloads/econet_ped/echonetpediatric/pediatric_echo_avi/pediatric_echo_avi/A4C/Videos/'
path_to_tracings = '/home/paul/Downloads/econet_ped/echonetpediatric/pediatric_echo_avi/pediatric_echo_avi/A4C/VolumeTracings.csv'
save_path = '/home/paul/Downloads/echonet_ped_preprocessed/'

preprocess_echonet_ped(path_to_csv, path_to_tracings, path_to_videos, save_path)