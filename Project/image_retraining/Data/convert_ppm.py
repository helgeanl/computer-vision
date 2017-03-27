from PIL import Image

foldername = 'speed_sone_over_80'
foldername_jpg = 'speed_sone_over_80_jpg'

for i in range(13):
    for j in range(30):
        if i < 10:
            if j < 10:
                filename    = foldername+'/0000'+str(i)+'_0000'+str(j)+'.ppm'
                filenameJPG = foldername_jpg+'/0000'+str(i)+'_0000'+str(j)+'.jpg'
            else:
                filename = foldername+'/0000'+str(i)+'_000'+str(j)+'.ppm'
                filenameJPG = foldername_jpg+'/0000'+str(i)+'_000'+str(j)+'.jpg'
        else:
            if j < 10:
                filename    = foldername+'/000'+str(i)+'_0000'+str(j)+'.ppm'
                filenameJPG = foldername_jpg+'/000'+str(i)+'_0000'+str(j)+'.jpg'
            else:
                filename = foldername+'/000'+str(i)+'_000'+str(j)+'.ppm'
                filenameJPG = foldername_jpg+'/000'+str(i)+'_000'+str(j)+'.jpg'
        im = Image.open(filename)
        im.save(filenameJPG)
