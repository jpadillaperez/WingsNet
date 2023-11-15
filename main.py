import os
import numpy as np
from WingsNet import get_model
import torch
import nibabel as nib
import datetime
from skimage import measure
from utils.paths import input_folder, output_folder, file_extension

#------------------------------------------------------------------------------------------
#Get device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

#------------------------------------------------------------------------------------------
#Import model
_ , model = get_model()
model.load_state_dict(torch.load('WingsNet_GUL.ckpt')['state_dict'])
model = model.to(device)
model.eval()

#------------------------------------------------------------------------------------------
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.startswith('.') or file.startswith('@'):
            continue
        if file.endswith(file_extension):
            file_path = os.path.join(root, file)

            #----------------------------------------------------------------------------------------------
            # Create output folder in the same structure as the original folder tree
            current_output_folder = os.path.join(output_folder, os.path.relpath(root, input_folder))
            print("Output folder: ", current_output_folder)
            os.makedirs(current_output_folder, exist_ok=True)

            #----------------------------------------------------------------------------------------------
            # Input file
            img = nib.load(file_path)
            data = img.get_fdata()
            input_data = torch.from_numpy(data).float()
            input_data = input_data.to(device)

            #----------------------------------------------------------------------------------------------
            # Prediction

            #sliding window
            cube_size = 128
            step = 64

            pred = np.zeros(input_data.shape)
            pred_num = np.zeros(input_data.shape)
            xnum = (input_data.shape[0]-cube_size)//step + 1 if (input_data.shape[0]-cube_size)%step==0 else (input_data.shape[0]-cube_size)//step + 2
            ynum = (input_data.shape[1]-cube_size)//step + 1 if (input_data.shape[1]-cube_size)%step==0 else (input_data.shape[1]-cube_size)//step + 2
            znum = (input_data.shape[2]-cube_size)//step + 1 if (input_data.shape[2]-cube_size)%step==0 else (input_data.shape[2]-cube_size)//step + 2
            for xx in range(xnum):
                xl = step*xx
                xr = step*xx + cube_size
                if xr > input_data.shape[0]:
                    xr = input_data.shape[0]
                    xl = input_data.shape[0]-cube_size
                for yy in range(ynum):
                    yl = step*yy
                    yr = step*yy + cube_size
                    if yr > input_data.shape[1]:
                        yr = input_data.shape[1]
                        yl = input_data.shape[1] - cube_size
                    for zz in range(znum):
                        zl = step*zz
                        zr = step*zz + cube_size
                        if zr > input_data.shape[2]:
                            zr = input_data.shape[2]
                            zl = input_data.shape[2] - cube_size
                        
                        input_window = input_data[xl:xr, yl:yr, zl:zr].unsqueeze(0).unsqueeze(0)
                        _, p = model(input_window)
                        p = torch.sigmoid(p)
                        p = p.cpu().detach().numpy()
                        pred[xl:xr,yl:yr,zl:zr] += p.squeeze()
                        pred_num[xl:xr,yl:yr,zl:zr] += 1
                        
            pred = np.array(np.round(pred/pred_num), dtype=np.int8)
            pred = np.squeeze(pred)
            pred = np.where(pred > 0.5, 1, 0)

            #----------------------------------------------------------------------------------------------
            # Save output file
            #change type of the segmentation in header
            img.header.set_data_dtype(np.uint8)
            pred_nii = nib.Nifti1Image(pred, img.affine, img.header)
            nib.save(pred_nii, os.path.join(current_output_folder, file[:-len(file_extension)] + "_airway.nii.gz"))
            print("Saved file: " + os.path.join(current_output_folder, file[:-len(file_extension)] + "_airway.nii.gz"))
