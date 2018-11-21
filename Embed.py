
import cv2
from utils import torch_to_img
import numpy

def Embed(output, whole_size, res, No):

	x,y,w,h = int(res[0]),int(res[1]),int(res[2]),int(res[3])
	print('x:{},y:{},w:{},h:{}').format(x,y,w,h)
	part_size = (w,h)
	output = output.cpu().detach().numpy()*255
	output = cv2.resize(output, part_size, interpolation=cv2.INTER_AREA)
	assert output.shape[0] == h, 'resize shape wrong {} and {}'.format(output.shape[0],h)
	assert output.shape[1] == w, 'resize shape wrong {} and {}'.format(output.shape[1],w)
	final = numpy.zeros(whole_size)

	start_y = max(0,y)
	start_x = max(0,x)
	end_y = min(whole_size[0], y+h)
	end_x = min(whole_size[1], x+w)

	final[start_y:end_y, start_x:end_x] = output[start_y-y: h-(y+h-end_y), start_x-x: w-(x+w-end_x)]

	cv2.imwrite('./output_segs/'+str(No)+'.png',final)
	print('write No.'+str(No)+' frame')
