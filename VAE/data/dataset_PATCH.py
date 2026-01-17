import os
import glob
from monai.data import Dataset, DataLoader, CacheDataset
from . import config


class MioDataset(Dataset):

    def __init__(self, opt, root_dir):
        self.opt = opt
        self.root_dir = root_dir
        self.classes = ['matched', 'nuove', 'scomparse']
        self.data_list = self._load_files()

    def _load_files(self):
        data_list = []

        # Percorsi base
        images_root = os.path.join(self.root_dir, 'images')
        mask_root = os.path.join(self.root_dir, 'masks')

        for class_name in self.classes:
            class_img_dir = os.path.join(images_root, class_name)
            class_mask_dir = os.path.join(mask_root, class_name)

            if not os.path.isdir(class_img_dir):
                print(f"Attenzione: Cartella {class_img_dir} non trovata.")
                continue

            if not os.path.isdir(class_mask_dir):
                print(f"Attenzione: Cartella {class_mask_dir} non trovata.")
                continue


            ct_files = sorted(glob.glob(os.path.join(class_img_dir, "*_ct.nii.gz")))

            for ct_path in ct_files:
                filename = os.path.basename(ct_path)
                pid = filename.split('_ct')[0]

                # Esempio filename: 013d407166_0_lesion12_baseline_ct.nii.gz

                mask_filename = filename.replace('_ct.nii.gz', '_mask.nii.gz')
                mask_path = os.path.join(class_mask_dir, mask_filename)


                data_list.append({
                    'image': ct_path,
                    'mask': mask_path,
                    'class_name': class_name,
                    'patient_id': pid
                })


        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # Nota: Se usi CacheDataset di MONAI, questo __getitem__ potrebbe non essere chiamato direttamente
        # nel loop di training se passi `data_list` direttamente al CacheDataset,
        # ma è utile averlo per debug o uso standard.
        return self.data_list[index]


def CreateDataloader(opt, shuffle=True, num_workers=0, drop_last=False, cache=True):

    if not os.path.exists(opt.dataroot):
        print(f"[ERROR] Percorso non trovato: {opt.dataroot}")
        return None


    dataset_helper = MioDataset(opt, opt.dataroot)
    data_list = dataset_helper.data_list

    if len(data_list) == 0:
        print("[ERROR] Nessun dato trovato. Controlla i percorsi.")
        return None


    transforms = config.train_transforms if opt.phase == 'train' else config.test_transforms

    if cache:
        ds = CacheDataset(
            data=data_list,
            transform=transforms
        )
    else:
        ds = Dataset(
            data=data_list,
            transform=transforms
        )

    data_loader = DataLoader(
        ds,
        batch_size=opt.batchSize,
        num_workers=num_workers,
        drop_last=drop_last,
        shuffle=shuffle,
        pin_memory=True
    )

    return data_loader