
# the size of warped map is set to 16
import torch
from torch.autograd import Variable





def bilinear(feature, res, BSrchFlag):
	#feature = feature.cpu()
	if BSrchFlag==0:
		spatial_scale = 0.125
	else:
		spatial_scale = 0.125

	pooled_height = 16.0
	pooled_width = 16.0
	batch = feature.size(0)
	channel = feature.size(1)
	height = feature.size(2)  #24
	width = feature.size(3)   #24

	assert batch == 1, 'batch is not 1'
	assert height>0 , 'height not positive'
	assert width>0 , 'width not positive'

	out = Variable(torch.empty(batch,channel,int(pooled_height),int(pooled_width)).cuda(), requires_grad = False)

	res[0] = max(0,res[0])
	res[1] = max(0,res[1])

	roi_start_w = (res[0] * spatial_scale)
	roi_start_h = (res[1] * spatial_scale)
	roi_width = res[2] * spatial_scale
	roi_height = res[3] * spatial_scale

	assert roi_start_w>=0 , 'roi_start_w not positive'
	assert roi_start_h>=0 , 'roi_start_h not positive'
	assert roi_width>=0 , 'roi_width not positive'
	assert roi_height>=0 , 'roi_height not positive'

	bin_size_h = roi_height / pooled_height
	bin_size_w = roi_width / pooled_width

	for ph in range(int(pooled_height)):
		for pw in range(int(pooled_width)):
			ih = roi_start_h + ph * bin_size_h
			iw = roi_start_w + pw * bin_size_w

			h_low = int(ih)
			w_low = int(iw)

			if h_low >= height-1:
				h_high = h_low = height-1
				ih = h_low
			else:
				h_high = h_low +1

			if w_low >= width-1:
				w_high = w_low = width-1
				iw = w_low
			else:
				w_high = w_low +1

			lh = ih - float(h_low)
			lw = iw - float(w_low)
			hh = 1.0 - lh
			hw = 1.0 - lw

			out[0,:,ph,pw] = feature[0,:,h_low,w_low]*hh*hw + feature[0,:,h_low,w_high]*hh*lw \
									+ feature[0,:,h_high,w_low]*lh*hw + feature[0,:,h_high,w_high]*lh*lw

	#out = out.cuda()

	return out

