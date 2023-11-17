import os
import shutil
for i in os.listdir('../'):
    if '.' not in i and i[-1].isdigit():
        print('../'+i+'/results/'+'REFLECTANCE_'+i+'.dat')
        print('../'+i+'/results/'+'REFLECTANCE_'+i+'.hdr')
        shutil.copyfile('../'+i+'/results/'+'REFLECTANCE_'+i+'.dat', './dat/'+'REFLECTANCE_'+i+'.dat')
        shutil.copyfile('../'+i+'/results/'+'REFLECTANCE_'+i+'.hdr', './hdr/'+'REFLECTANCE_'+i+'.hdr')
        # for j in os.listdir(os.path.join('..', i)):
        #     if '.png' in j:
        #         print('../'+i+'/'+j)
        #         shutil.copyfile('../'+i+'/'+j, j)