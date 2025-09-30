from data_converter.nuscenes_converter_seg import  create_nuscenes_infos

data_root = "/scratch1/ayushgoy/nuscenes_extracted"

if __name__ == '__main__':
    # Training settings
    create_nuscenes_infos(data_root ,'nuscenes_dataset')

