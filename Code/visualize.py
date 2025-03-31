import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import re

def get_patient_ids(dataset_path):
    patient_ids = []
    for folder in os.listdir(dataset_path):
        if folder.startswith("patient"):
            try:
                patient_id = int(folder.replace("patient", ""))
                patient_ids.append(patient_id)
            except ValueError:
                continue  
    return sorted(patient_ids)

def visualize_all_slices(dataset_type):
    dataset_path = os.path.join("cropped_slices_processed", dataset_type)
    patient_ids = get_patient_ids(dataset_path)  
    
    for patient_id in patient_ids:
        patient_folder = os.path.join("cropped_slices_processed", dataset_type, f"patient{patient_id:03d}")
        nii_files = os.listdir(patient_folder)
        
        frame_numbers = [
            int(re.search(r'frame(\d+)', f).group(1))
            for f in nii_files if re.search(r'frame(\d+)', f) and "frame01" not in f
        ]

        files = {
            "ED": f"patient{patient_id:03d}_ED_processed.nii.gz",
            "ED_gt": f"patient{patient_id:03d}_ED_gt_processed.nii.gz",
            "ES": f"patient{patient_id:03d}_ES_processed.nii.gz",
            "ES_gt": f"patient{patient_id:03d}_ES_gt_processed.nii.gz",
        }


        for phase in ["ED", "ES"]:
            if not os.path.exists(os.path.join(patient_folder, files[phase])):
                continue

            print(f"\n📊 Visualizing Patient {patient_id:03d}")
            image = nib.load(os.path.join(patient_folder, files[phase])).get_fdata()
            mask_path = os.path.join(patient_folder, files[phase + "_gt"])
            has_mask = os.path.exists(mask_path)
            mask = nib.load(mask_path).get_fdata() if has_mask else None

            num_slices = image.shape[2]
            for z in range(num_slices):
                fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                axs[0].imshow(image[:, :, z], cmap='gray')
                axs[0].set_title(f"Slice {z} after ROI cropping")
                axs[0].axis("off")

                if has_mask and np.any(mask[:, :, z] > 0):
                    axs[1].imshow(mask[:, :, z], cmap='jet', vmin=0, vmax=3)
                    axs[1].set_title("Ground Truth")
                else:
                    axs[1].text(0.5, 0.5, 'No GT available', ha='center', va='center', fontsize=10)
                    axs[1].set_title("No Ground Truth")

                axs[1].axis("off")

                plt.tight_layout()
                plt.show()

visualize_all_slices("training")