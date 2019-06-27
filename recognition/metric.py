import numpy as np
import mxnet as mx

class AccMetric(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(AccMetric, self).__init__(
        'acc', axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []
    self.count = 0

  def update(self, labels, preds):
    self.count+=1
    label = labels[0]
    # print('label.shape:',str(len(labels)))
    # print(label)
    pred_label = preds[1]
    # print(len(pred_label))
   ### print('ACC', label.shape, pred_label.shape)###(1, num_classes)
    if pred_label.shape != label.shape:
        pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
    pred_label = pred_label.asnumpy().astype('int32').flatten()
#    print('pred_Label',pred_label.shape)###1
    label = label.asnumpy()
    if label.ndim==2:
      label = label[:,0]
    label = label.astype('int32').flatten()
    assert label.shape==pred_label.shape
#    print('ACC',pred_label, label)  
    self.sum_metric += (pred_label.flat == label.flat).sum()  ####right samples 
    self.num_inst += len(pred_label.flat)  #### general samples

class LossValueMetric(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(LossValueMetric, self).__init__(
        'lossvalue', axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []

  def update(self, labels, preds):
    #label = labels[0].asnumpy()
    pred = preds[-1].asnumpy()
#    pred_dis =  preds[-2].asnumpy()
#    print(pred.shape)###1
    #print('in loss', pred.shape)
    #print(pred)
    loss = pred[0]
#    print('pred:', str(pred[0]))
    self.sum_metric += loss
    self.num_inst += 1.0
    #gt_label = preds[-2].asnumpy()
    #print(gt_label)
