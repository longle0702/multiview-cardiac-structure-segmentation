import os
import numpy as np 
import matplotlib.pyplot as plt
import torch 
from torch.utils.data import DataLoader
import nibabel as nib

from src.model.unet import UNet3D
from src.dataset.acdc import ACDC
from src.dataset.acdc_processed import ACDCProcessed
from src.dataset.just_to_test import JustToTest
from src.metrics import calculate_multiclass_dice


if __name__ == "__main__": 
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    device = "cuda"
    model_path = "model/unet3d_8_best.pth"
    print(f"Using {model_path}")

    model = UNet3D(out_channels=3, binary = True).to(device)

    model.load_state_dict(torch.load(model_path,map_location = torch.device(device)))

    test_dataset = ACDCProcessed("cropped_slices_processed/testing", is_testset=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dices = []
    class1_dice = []
    class2_dice = []
    class3_dice = []

    with torch.no_grad(): 
        for i,data in enumerate(test_dataloader): 
            img, gt, img_path = data
            img = img.to(torch.float).to(device)
            gt = gt.to(device)

            output = model(img)
            # output = (output > 0.5).to(torch.long)

            dice, dice_list = calculate_multiclass_dice(
                output.squeeze(), 
                gt.squeeze(), 
                num_class=3
            )
            
            class1_dice.append(dice_list[0])
            class2_dice.append(dice_list[1])
            class3_dice.append(dice_list[2])
            dices.append(dice)  
            
            print(f"Dice: {dice:.3f} {np.round(dice_list, 3)} ")

            # Save output to npy file
            basename = os.path.basename(img_path[0])
            patient_id = basename[7:10]
            frame_id = basename[11:18]

            if ("ED" in frame_id): 
                is_ed = True
            else: 
                is_ed = False

            for layer in range(output.size(2)): 
                output_numpy = output.squeeze().permute(2,3,0,1).cpu().numpy()[:,:,:,layer]
                slice = layer+1
                
                if (is_ed):
                    file_name = "3d_" + "patient" + str(patient_id) + "_phaseED_" + "slice" + str(slice) + ".npy"
                else: 
                    file_name = "3d_" + "patient" + str(patient_id) + "_phaseES_" + "slice" + str(slice) + ".npy"

                np.save(os.path.join("output_numpy", file_name), output_numpy)


                      

    mean_dice_class1 = np.mean(class1_dice)
    mean_dice_class2 = np.mean(class2_dice)
    mean_dice_class3 = np.mean(class3_dice)

    print(f"Mean Dice of class 1: {mean_dice_class1:.3f}") 
    print(f"Mean Dice of class 2: {mean_dice_class2:.3f}", )
    print(f"Mean Dice of class 3: {mean_dice_class3:.3f}", )
    print(f"Mean Dice: ", np.round(np.mean(dices), 3))

