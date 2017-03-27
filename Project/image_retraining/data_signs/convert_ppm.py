from PIL import Image
import os.path

origin_folder = 'oldPPM/'

for k in range(43):
    dest_directory = str(k).zfill(5)
    if not os.path.exists(dest_directory):
      os.makedirs(dest_directory)
    for i in range(7):
        for j in range(30):
            str(1).zfill(2)
            filename    = str(i).zfill(5)+'_'+str(j).zfill(5)+'.ppm'
            filenameJPG = str(i).zfill(5)+'_'+str(j).zfill(5)+'.jpg'
            im = Image.open(origin_folder+dest_directory+'/'+filename)
            im.save(dest_directory+'/'+filenameJPG)
