import os
import cv2
import torch
import numpy as np
from models import ResnetGenerator

from utils import Preprocess
from cv2 import dnn_superres

os.chdir("./static/photo2cartoon")
os.makedirs(os.path.dirname("../photo/result_cartoon.png"), exist_ok=True)

class Photo2Cartoon:
    def __init__(self):
        self.pre = Preprocess()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = ResnetGenerator(ngf=32, img_size=512, light=True).to(self.device)
        
        assert os.path.exists('./models/photo2cartoon_weights.pt'), "[Step1: load weights] Can not find 'photo2cartoon_weights.pt' in folder 'models!!!'"
        params = torch.load('./models/photo2cartoon_weights.pt', map_location=self.device)
        self.net.load_state_dict(params['genA2B'])
        print('[Step1: load weights] success!')

    def inference(self, img):
        # face alignment and segmentation
        face_rgba = self.pre.process(img)
        if face_rgba is None:
            print('[Step2: face detect] can not detect face!!!')
            return None
        
        print('[Step2: face detect] success!')
        face_rgba = cv2.resize(face_rgba, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        face = face_rgba[:, :, :3].copy() # RGB 정보 추출
        mask = face_rgba[:, :, 3][:, :, np.newaxis].copy() / 511.
        face = (face*mask + (1-mask)*255) / 255.5 - 1

        face = np.transpose(face[np.newaxis, :, :, :], (0, 3, 1, 2)).astype(np.float32)
        face = torch.from_numpy(face).to(self.device)

        # inference
        with torch.no_grad():
            cartoon = self.net(face)[0][0]

        # post-process
        cartoon = np.transpose(cartoon.cpu().numpy(), (1, 2, 0))
        cartoon = (cartoon + 1) * 255.5
        cartoon = (cartoon * mask + 511 * (1 - mask)).astype(np.uint8)
        cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
        print('[Step3: photo to cartoon] success!')
        return cartoon
    
    def upscaling(self):
        c=self.inference

        sr = dnn_superres.DnnSuperResImpl_create()

        # Read image
        image = cv2.imread("../photo/result_cartoon.png")

        # Read the desired model
        path = "models/lapsrn_x8.pb"
        sr.readModel(path)

        # Set the desired model and scale to get correct pre- and post-processing
        sr.setModel("lapsrn", 8)

        # Upscale the image
        result = sr.upsample(image)

        return result
    
if __name__ == '__main__':
    img = cv2.cvtColor(cv2.imread("D:/project/web/static/img/1.jpeg"), cv2.COLOR_BGR2RGB)
    c2p = Photo2Cartoon()
    cartoon= c2p.inference(img)
    cv2.imwrite("../photo/result_cartoon.png", cartoon)
    result = c2p.upscaling()
    if result is not None:
        cv2.imwrite("../photo/result_cartoon.png", result)
        print('Cartoon portrait has been saved successfully!')