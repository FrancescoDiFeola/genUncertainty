import os
from monai.data import Dataset, DataLoader, CacheDataset
from . import config
#import data.config





class MioDataset(Dataset):

    def __init__(self, opt, folder_paths):
        self.opt = opt
        self.folder_paths = folder_paths
        self.files_A = self._load_files()


    def _load_files(self):
        files_A = []

        for patient_folder_path in self.folder_paths:
            pid = os.path.basename(patient_folder_path)

            try:
                subdirs = [d for d in os.listdir(patient_folder_path)
                           if os.path.isdir(os.path.join(patient_folder_path, d))]
            except OSError as e:
                print(f"Impossibile leggere la cartella del paziente {pid}: {e}")
                continue

            relevant_folders = sorted([
                d for d in subdirs if d.startswith('baseline') or d.startswith('followup')
            ])

            if not relevant_folders:
                print(f"Nessuna cartella 'baseline...' o 'followup...' trovata per {pid}")
                continue

            for folder_name in relevant_folders:
                current_ct_path = os.path.join(patient_folder_path, folder_name)

                try:
                    volume_files = sorted([
                        f for f in os.listdir(current_ct_path)
                        if "ct" in f
                           and "mask" not in f
                           and (f.endswith(".nii") or f.endswith(".nii.gz"))
                    ])
                except OSError as e:
                    print(f"Impossibile leggere la cartella {current_ct_path}: {e}")
                    continue

                for file_name in volume_files:
                    file_path = os.path.join(current_ct_path, file_name)

                    if os.path.exists(file_path):
                        files_A.append(file_path)


        return files_A


    def __getitem__(self, index):
        img_path_A = self.files_A[index]


        volume_name = os.path.basename(img_path_A).replace(".nii.gz", "")

        folder_path = os.path.dirname(img_path_A)
        folder_name = os.path.basename(folder_path)

        patient_folder_path = os.path.dirname(folder_path)
        patient_id = os.path.basename(patient_folder_path)

        a_name = f"{patient_id}_{folder_name}_{volume_name}"

        file_dict = {
            'A': img_path_A,
        }

        return {**file_dict,
                'A_paths': img_path_A,
                'a_name': a_name}



    def __len__(self):
        return len(self.files_A)



def CreateDataloader(opt, shuffle=True, num_workers=0, drop_last=False,cache=True):

    folder_paths = [
        os.path.join(opt.dataroot, d)
        for d in os.listdir(opt.dataroot)
        if os.path.isdir(os.path.join(opt.dataroot, d))
    ]

    if not folder_paths:
        print(f"[ERROR] Nessuna sottocartella trovata in {opt.dataroot}")
        return None


    mio_dataset = MioDataset(opt, folder_paths)
    #data_list = [mio_dataset[i] for i in range(len(mio_dataset))]

    # Decide se utilizzare CacheDataset o Dataset in base al valore di 'cache'
    if cache:
        ds = CacheDataset(
            data=mio_dataset,
            transform=config.train_transforms if opt.phase == 'train' else config.test_transforms)
    else:
        ds = Dataset(
            data=mio_dataset,
            transform=config.train_transforms if opt.phase == 'train' else config.test_transforms)

    data_loader = DataLoader(
        ds,
        batch_size=opt.batchSize,
        num_workers=num_workers,
        drop_last=drop_last,
        shuffle=shuffle,
        pin_memory=True
    )

    return data_loader

