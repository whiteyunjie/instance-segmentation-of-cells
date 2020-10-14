import cv2
import numpy as np
from sklearn.cluster import KMeans 
def watershed_process(mask):
    kernel = np.ones((3,3),np.uint8)
    sure_bg = cv2.dilate(mask,kernel,iterations=3)
    dist_transform = cv2.distanceTransform(mask,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.592*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 1] = 0
    rgb = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    markers2 = cv2.watershed(rgb,markers)
    newmark=markers2
    newmark[markers2==-1] = 1
    newmark = newmark-1
    return newmark

def kluster_proess(mask):
    #maxval,pred_img,_,_ = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    pred_img = watershed_process(mask)
    zonelabels = np.unique(pred_img)
    imgpro = np.zeros(mask.shape,np.uint8)
    curnum = 0
    for i in range(1,len(zonelabels)):
        zone = np.zeros(mask.shape,np.uint8)
        zone[pred_img==i]=1
        index = np.argwhere(pred_img==i)
        # plt.imshow(zone)
        # plt.show()
        #erode
        kernel = np.ones((3,3),np.uint8)
        img_erode = cv2.erode(zone,kernel,iterations = 20)
        # plt.imshow(img_erode)
        # plt.show()

        #clusters
        maxval,curzone,_,centorids = cv2.connectedComponentsWithStats(img_erode, 4, cv2.CV_32S)
        curlabels = np.unique(curzone)
        kclasses = len(curlabels)-1

        flags = centorids[1:,:]
        if len(flags)>0:
            dataxy = index.astype(np.float32)
            clf = KMeans(n_clusters=kclasses,init=flags,n_init=1,tol=1e-6)
            clf.fit(dataxy)
            clf.labels_ += 1 #最小标签是0

            clf.labels_ += curnum #防止标签重复
            for j in range(len(index)):
                imgpro[index[j,0]][index[j,1]] = clf.labels_[j]
            curnum = np.max(clf.labels_)
        else:
            for j in range(len(index)):
                imgpro[index[j,0]][index[j,1]] = curnum + 1
            curnum = 1
            
        #print(curnum)
    #plt.imshow(imgpro)
    #plt.show()
    return imgpro