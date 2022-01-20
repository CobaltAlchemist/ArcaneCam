import pyvirtualcam
import numpy as np
import matplotlib.pyplot as plt
import cv2
import multiprocessing
from torchvision import transforms
from facenet_pytorch import MTCNN
import torch, PIL
import torch.nn.functional as F
import time
from tqdm.auto import tqdm

def detect(mtcnn, img):
	# Detect faces
	batch_boxes, batch_probs, batch_points = mtcnn.detect(img, landmarks=True)
	# Select faces
	if not mtcnn.keep_all:
		batch_boxes, batch_probs, batch_points = mtcnn.select_boxes(
				batch_boxes, batch_probs, batch_points, img, method=mtcnn.selection_method
		)

	return batch_boxes, batch_points
	
def get_ratio(mtcnn, _img, max_res=1_500_000, target_face=256, fix_ratio=0, max_upscale=2, VERBOSE=False):
	boxes = None
	h, w, _ = _img.shape
	subscale = h / 240
	new_w = makeEven(int(w / subscale))
	new_h = makeEven(int(h / subscale))
	smolimg = cv2.resize(_img, (new_w, new_h))
	boxes, _ = detect(mtcnn, smolimg)
	if VERBOSE: print('boxes',boxes)
	
	ratio = 2 #initial ratio

	#scale to desired face size
	if (boxes is not None):
		if len(boxes)>0:
			ratio = target_face/max(boxes[0][2:]-boxes[0][:2])
			ratio = min(ratio / subscale, max_upscale)
			if VERBOSE: print('up by', ratio)
	if fix_ratio>0:
		if VERBOSE: print('fixed ratio')
		ratio = fix_ratio

	w*=ratio
	h*=ratio

	#downscale to fit into max res 
	res = w*h
	if res > max_res:
		ratio = pow(res/max_res,1/2); 
		if VERBOSE: print(ratio)
	return ratio
	
# my version of isOdd, should make a separate repo for it :D
def makeEven(_x):
	return _x if (_x % 2 == 0) else _x+1
	
def scale_image(_img, ratio=None, VERBOSE=False):
	h, w, _ = _img.shape
	w=int(w/ratio)
	h=int(h/ratio)
	w = makeEven(int(w))
	h = makeEven(int(h))
	return cv2.resize(_img, (w, h))
	
	
def ratio_thread(p):
	mtcnn = MTCNN(image_size=256, margin=80, device='cpu').eval()
	print("Starting ratio thread")
	p.send((1., 0))
	while True:
		pkg = p.recv()
		if pkg is None:
			break
		img, target_face, max_res, max_upscale = pkg
		start = time.time()
		with torch.no_grad():
			ratio = get_ratio(mtcnn, img, max_res=max_res, target_face=target_face, fix_ratio=0, max_upscale=max_upscale, VERBOSE=False)
		latency = (time.time() - start) * 1000
		p.send((ratio, latency))
		
class Arcanizer:
	def __init__(self, max_img_size = (1, 1)):
		self.model = torch.jit.load('ArcaneGANv0.3.jit',map_location='cuda').eval()
		self.model(torch.zeros((1, 3, *max_img_size)).half().cuda())
		self.ratio = 1.
		self.maxh, self.maxw = max_img_size
		self.log = {}
		means = [0.485, 0.456, 0.406]
		stds = [0.229, 0.224, 0.225]
		self.t_stds = torch.tensor(stds)[:,None,None].cuda()
		self.t_means = torch.tensor(means)[:,None,None].cuda()
		self.transforms = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(means,stds)])
			
	def _tensor2im(self, var):
		return var.mul(self.t_stds).add(self.t_means).mul(255.).clamp(0,255).permute(1,2,0)
		
	def __call__(self, input_image):
		h, w, _ = input_image.shape
		
		p = time.time()
		if self.ratio != 1.:
			scaled = scale_image(input_image, ratio=self.ratio)
		else:
			scaled = input_image
		sh, sw, _ = scaled.shape
		self.log['scaling'] = (time.time() - p) * 1000
		p = time.time()
		tform = self.transforms(scaled).unsqueeze(0).half().cuda()
		tform = F.pad(tform, (0, self.maxw - sw, 0, self.maxh - sh))
		self.log['tform'] = (time.time() - p) * 1000
		
		p = time.time()
		
		with torch.no_grad():
			img = self.model(tform)[0]
		self.log['model'] = (time.time() - p) * 1000
		p = time.time()
		
		img = self._tensor2im(img)
		img = img[:sh, :sw]
		img = img.detach().cpu().numpy().astype('uint8')
		if w != sw:
			img = cv2.resize(img, (w, h))
		return img
		
	
if __name__ == "__main__":
	proc_res = (1024, 576)
	proc_res = (640, 360)
	proc_res = (1280, 720)
	cap_res = (1280, 720)
	#arc = Arcanizer([x*2 for x in proc_res[::-1]])
	arc = Arcanizer(proc_res[::-1])
	
	p, p2 = multiprocessing.Pipe(True)
	#worker = multiprocessing.Process(target=ratio_thread, args=(p2,))
	#worker.start()
	
	cv2.destroyAllWindows()
	cap = cv2.VideoCapture(1)
	codec = 0x47504A4D  # MJPG
	cap.set(cv2.CAP_PROP_FPS, 30.0)
	cap.set(cv2.CAP_PROP_FOURCC, codec)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_res[0])
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_res[1])
	ret, frame = cap.read()
	h, w, _ = frame.shape
	print(f"Starting with {w}x{h}")
	latency = 0.
	bar = tqdm()
	with pyvirtualcam.Camera(width=proc_res[0], height=proc_res[1], fps=30) as cam:
		try:
			while True:
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				frame = cv2.resize(frame, proc_res)
				if p.poll():
					arc.ratio, latency = p.recv()
					arc.ratio = max(arc.ratio, 1.)
					p.send((frame, 300, 4 * proc_res[0] * proc_res[1], 2))
				
				bar.set_postfix({**arc.log, 'mtcnn': latency, 'ratio': round(arc.ratio, 2), 'max_seen': (arc.maxw, arc.maxh)})
				arcane = arc(frame)
				
				cam.send(arcane)
				bar.update()
				ret, frame = cap.read()
				cam.sleep_until_next_frame()
		except KeyboardInterrupt:
			cap.release()
			
	cap.release()