import numpy as np
import os
root='/home/hankeji/Desktop/Adversarial Examples/'
new_pth='/home/hankeji/Desktop/domain adaptation/'
for root_pth, sub_pth, files in os.walk(root):
    for file in files:
        pth=os.path.join(root, file)
        length=len(pth)
        print ('*'*length)
        #print (str(file))
        if os.path.splitext(pth)[1]=='.npy':
            if 'Label' in str(pth):
                print (str(pth))
                tmp_npy = np.load(str(pth))
                if tmp_npy.shape[0]==10000:
                    relabel=[]
                    relabel=np.asarray(relabel, np.int)
                    for i in range(10000):
                        tmp_data=tmp_npy[i]
                        tmp_data=np.reshape(tmp_data, (-1, 10))
                        tmp_max=np.where(tmp_data==np.max(tmp_data, 1))
                        tmp_label=int(tmp_max[1])
                        relabel=np.append(relabel, tmp_label)
                    relabel=np.reshape(relabel, (-1,1))
                    print (relabel.shape)
                    np.save(pth, relabel)
                #cv2.imshow('First Image', tmp_npy[0])
                #cv2.waitKey(1000)
                #cv2.destroyAllWindows()

        print ('*' * length)
        print ('\n')