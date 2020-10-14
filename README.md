# instance-segmentation-of-cells
instance segmentation task. Big assignment of pattern recognition course

it's an instance segmentation task. However, I have finished it through methods of **semantic segmentation and post processing.**

### backbone:
  UNet


### post processing:
  watershed
  Connected domain
  clustering

### results:
![](https://github.com/whiteyunjie/instance-segmentation-of-cells/blob/main/clusters.png)
![](https://github.com/whiteyunjie/instance-segmentation-of-cells/blob/main/watershed_process.png)

### metrics:
Jaccard similarity

### usage:
#### train
  `python train.py`
evaluate training results
  `python trainjs.py`
#### test
  `python test.py`
#### others
**path**
dataset1：
	os.path.join('dataset1/train/',img) for img in os.listdir('dataset1/train/')
  
  os.path.join('dataset1/train_GT/SEG/',img) for img in os.listdir('dataset1/train_GT/SEG/')
  
dataset2:

	os.path.join('dataset2/train/',img) for img in os.listdir('dataset2/train/')
  
  os.path.join('dataset2/train_GT/SEG/',img) for img in os.listdir('dataset2/train_GT/SEG/')
  
post process:'connectedComponents','watershed'

imgsize：628 dataset1,500 dataset2

The data and weights are available if you want to try.
