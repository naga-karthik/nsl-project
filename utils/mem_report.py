import os,sys,humanize,psutil,GPUtil
import torch
import torchvision.models as models

def mem_report():
      print("CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))

      GPUs = GPUtil.getGPUs()
      for i, gpu in enumerate(GPUs):
            print('GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))
      return gpu.memoryTotal - gpu.memoryFree


# model = models.resnet18(pretrained=True)
# if torch.cuda.is_available():
#   model.cuda()

# mem_report()
