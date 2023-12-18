from sys import argv
from argparse import ArgumentParser
import os
import numpy as np
from tqdm import tqdm
import json
import cv2

from imageio.v2 import imread,imwrite
import math
import open3d as o3d



def parse_args():
    parser =  ArgumentParser(description="convert calib file to nerf format transforms.json")
    parser.add_argument("--calib", default="", help="specify calib file location")
    parser.add_argument("--output", default="transforms.json", help="output path")
    parser.add_argument("--points_out", default="points3d.ply", help="output point cloud")
    parser.add_argument("--create_alpha",action='store_true', help="create_images")
    parser.add_argument("--points_in", default="points3d.ply", help="location of folder with images")
    parser.add_argument("--imh", default=1440,type=int, help="height of image")
    parser.add_argument("--imw", default=810,type=int, help="width of image")
    parser.add_argument("--img_path", default="images/", help="location of folder with images")
    parser.add_argument("--mask_path", default="images/", help="location of folder with masks")
    parser.add_argument("--ext", default="exr", help="extention of input images")
    parser.add_argument("--lights_txt",
                        default="/CT/prithvi/work/studio-tools/LightStage/calibration/LSX_light_positions_aligned.txt",help="path to light dir (mm)")
    parser.add_argument("--lights_order", 
                        default="/CT/prithvi2/work/lightstage/LSX_python3_command/LSX/Data/LSX3_light_z_spiral.txt", help="order of lights")
    parser.add_argument("--scale", default=1,type=int, help="scale")
    args = parser.parse_args()
    return args

def read_light_dir(filename,scale=1):                          
     numbers_list = []                                          
     with open(filename, 'r') as file:                          
             for line in file:                          
                     numbers = line.strip().split()     
                     numbers = [float(n)/scale for n in numbers]        
                     numbers_list.append(numbers)          
     return numbers_list

def light_order(filename,lights):
    with open(filename) as light_file:
        z_spiral = np.asarray([int(line.strip())-1 for line in light_file if line.strip()]) 
    
    return np.asarray(lights)[z_spiral].tolist()
          


def read_calib(calib_dir):
    INTR = []
    EXTR = []   
    with open(calib_dir, 'r') as fp:
        while True:
            text = fp.readline()
            if not text:
                break

            elif "extrinsic" in text:
                lines = []
                for i in range(3):
                    line = fp.readline()
                    lines.append([float(w) for w in line.strip('#').strip().split()])
                lines.append([0, 0, 0, 1])
                extrinsic = np.array(lines).astype(np.float32)
                EXTR.append(np.linalg.inv(extrinsic))

            elif "intrinsic" in text:
                lines = []
                line = fp.readline()
                if "time" in line:
                    continue
                for i in range(3):
                    lines_3 = [float(w) for w in line.strip('#').strip().split()]
                    lines.append(lines_3)
                    line = fp.readline()
                intrinsic = np.array(lines).astype(np.float32)
                INTR.append(intrinsic)
               
    return INTR,EXTR    

def load_json(json_path):
    with open(json_path, 'r') as h:
        data = json.load(h)
    return data  

def save_json(data,json_path):
    with open(json_path,'w') as file:
        json.dump(data, file, sort_keys=True, indent=4)

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom


def central_point(out,transform_matrix_pcl):
    # find a central point they are all looking at
    print("computing center of attention...")
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for f in out["frames"]:
        mf = np.array(f["transform_matrix"])[0:3,:]
        for g in out["frames"]:
            mg = g["transform_matrix"][0:3,:]
            p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
            if w > 0.0001:
                totp += p*w
                totw += w
    totp /= totw
    print(totp) # the cameras are looking at totp
    for f in out["frames"]:
        f["transform_matrix"][0:3,3] -= totp
        f["transform_matrix"] = f["transform_matrix"].tolist()
    center_origin = np.asarray([0.5,0.1,0.5])
	# transform_matrix_pcl[0:3, 3] -= center_origin[0:3]
    transform_matrix_pcl[0:3, 3] -= totp[0:3]
    return out,transform_matrix_pcl,totp


def create_alpha(img_path,mask_path,alpha_path,scale,ext='exr'):
	# assuming all images and mask are following same naming convention. Breaks if it doesnt
    images = sorted(os.listdir(img_path))
    N = len(images)
    for i in tqdm(range(0,N)):
        mask = cv2.imread(os.path.join(mask_path,f'Cam_{str(i).zfill(2)}.jpg'),cv2.IMREAD_GRAYSCALE)
        h,w = mask.shape
        nh,nw = int(h/scale),int(w/scale)
        if ext == 'exr':
            img = imread(os.path.join(img_path,f'Cam_{str(i).zfill(2)}.exr'))
            mask[mask<100] = 0
            nimg = cv2.resize(img,(nw,nh),interpolation=cv2.INTER_AREA)
            nmask = cv2.resize(mask,(nw,nh),interpolation=cv2.INTER_AREA)/255.0
            alpha = nimg[...,:3] * (nmask)[...,np.newaxis]
            imwrite(os.path.join(alpha_path,f'Cam_{str(i).zfill(2)}.exr'),alpha.astype('float32'))
            png_alpha = np.clip(np.power(np.clip(alpha,0,None)+1e-5,0.45),0,1)*255 
            imwrite(os.path.join(alpha_path,f'Cam_{str(i).zfill(2)}.png'),png_alpha.astype('uint8'))


        elif ext == 'jpg':
            img = cv2.imread(os.path.join(img_path,f'Cam_{str(i).zfill(2)}.jpg'))
            nimg = cv2.resize(img,(nw,nh),interpolation=cv2.INTER_AREA)
            nmask = cv2.resize(mask,(nw,nh),interpolation=cv2.INTER_AREA)  
            alpha = cv2.merge((nimg,nmask))
            cv2.imwrite(os.path.join(alpha_path,f'Cam_{str(i).zfill(2)}.png'),alpha.astype('uint8'))
        
     
		

def parse_cameras(calib,path,imw,imh):
    intr,extr = read_calib(calib)
    transforms = {}
    transforms["aabb_scale"] = 16.0
    frames = []
    for idx in range(len(intr)):
        frame = {}
        frame['w'] = imw
        frame['h'] = imh
        frame['fl_x'] = np.asarray(intr[idx])[0,0] * imw
        frame['fl_y'] = np.asarray(intr[idx])[1,1] * imw
        frame['cx'] = np.asarray(intr[idx])[0,2] * imw
        frame['cy'] = np.asarray(intr[idx])[1,2] * imw
        frame['camera_angle_x'] = math.atan(float(imw) / (float(frame['fl_x'] * 2))) * 2
        frame['camera_angle_y'] = math.atan(float(imh) / (float(frame['fl_y'] * 2))) * 2
        frame['transform_matrix'] = extr[idx][[2,0,1,3],:]
        # frame['transform_matrix'] = extr[idx]
        frame['file_path'] = os.path.join(path,f'Cam_{str(idx).zfill(2)}')
        frames.append(frame)
    transforms["frames"] = frames
    return transforms

if __name__ == '__main__':
    
    args = parse_args()
    os.makedirs(args.output,exist_ok=True)
    img_folder = os.path.join(args.output,'images/')
    if args.create_alpha:
        os.makedirs(img_folder,exist_ok=True)
        create_alpha(args.img_path,args.mask_path,img_folder,args.scale,args.ext)

    sensor_x,sensor_y = 10.0000, 17.777
    transforms = parse_cameras(args.calib,img_folder,args.imw,args.imh)
    
    pcd = o3d.io.read_point_cloud(args.points_in)
    transform_matrix_pcl = np.eye(4,dtype='float32')
    # Rotation of axis
    transform_matrix_pcl = transform_matrix_pcl[[2,0,1,3],:]
    transforms,transforms_pcd,center = central_point(transforms,transform_matrix_pcl)
    # pcd.scale(3, center=pcd.get_center())
    pcd.transform(transforms_pcd)
    lights = {}
    usc_lights = np.asarray(read_light_dir(args.lights_txt,scale=1000))[:,[2,0,1]]
    lights['light_dir'] = [(direction - center).tolist() for direction in usc_lights]
    lights['light_dir'] = light_order(args.lights_order,lights['light_dir'])
    save_json(transforms,os.path.join(args.output,'transforms_train.json'))
    save_json(transforms,os.path.join(args.output,'transforms_test.json'))
    save_json(lights,os.path.join(args.output,'light_dir.json'))
    point_cloud_out = os.path.join(args.output,'points3d.ply')
    o3d.io.write_point_cloud(point_cloud_out, pcd)    
    