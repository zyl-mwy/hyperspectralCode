import os
import shutil
for i in os.listdir('../'):
    if '.' not in i and i[-1].isdigit():
        for j in os.listdir(os.path.join('..', i)):
            if '.png' in j:
                print('../'+i+'/'+j)
                shutil.copyfile('../'+i+'/'+j, j)