import numpy as np


def get_onelabelmap(data,label):
    index = np.argwhere(data==label)
    sam = np.zeros(data.shape)
    if len(index) is not 0:
        for i in range(len(index)):
            sam[index[i][0]][index[i][1]] = 1
    sam = sam.astype(np.int32)#注意类型转换，否则无法进行与运算
    return sam
def get_fit_GTimgs(cell_pred,label,cell_gt):
    #对于每个预测标签细胞，求与其匹配的gt细胞
    fit_lists = []
    Jaccard_lists = []
    labels = np.unique(cell_pred)
    cell_gt_map = get_onelabelmap(cell_gt,label)
    if len(labels) is not 0:
        for i in range(1,len(labels)):
            cell_pred_map = get_onelabelmap(cell_pred,labels[i])
            andmap=cell_pred_map&cell_gt_map
            if np.sum(andmap)>0.5*np.sum(cell_gt_map):
                a = np.sum(cell_pred_map&cell_gt_map)
                b = np.sum(cell_pred_map|cell_gt_map)
                fit_lists.append(labels[i])
                Jaccard_lists.append(a/b)
    return fit_lists,Jaccard_lists
def Jaccard_eval(cell_pred,cell_gt):
    #Jaccard相似度评估
    labels_gt = np.unique(cell_gt)
    JS_val = np.zeros((len(labels_gt)-1,))
    for i in range(1,len(labels_gt)):
        fit_lists,Jaccard_lists = get_fit_GTimgs(cell_pred,labels_gt[i],cell_gt)
        if len(fit_lists) is not 0:
             JS_val[i-1] = np.max(Jaccard_lists)
    return JS_val,np.mean(JS_val)