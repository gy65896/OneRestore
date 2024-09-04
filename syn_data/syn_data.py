import os, argparse, cv2, random
import numpy as np
from skimage import exposure

def guideFilter(I, p, winSize, eps):
    mean_I = cv2.blur(I, winSize)
    mean_p = cv2.blur(p, winSize)
    mean_II = cv2.blur(I * I, winSize)
    mean_Ip = cv2.blur(I * p, winSize)
    var_I = mean_II - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.blur(a, winSize)
    mean_b = cv2.blur(b, winSize)
    q = mean_a * I + mean_b
    return q

def syn_low(img, light, img_gray, light_max=3,
            light_min=2, noise_max=0.08, noise_min=0.03):
    light = guideFilter(light, img_gray,(3,3),0.01)[:, :, np.newaxis]
    n = np.random.uniform(noise_min, noise_max)
    R = img / (light + 1e-7)
    L = (light + 1e-7) ** np.random.uniform(light_min, light_max)
    return np.clip(R * L + np.random.normal(0, n, img.shape), 0, 1)

def syn_haze(img, depth, beta_max=2.0, beta_min=1.0, A_max=0.9, A_min=0.6,
                 color_max=0, color_min=0):
    beta = np.random.rand(1) * (beta_max - beta_min) + beta_min
    t = np.exp(-np.minimum(1 - cv2.blur(depth,(22,22)),0.7) * beta)
    A = np.random.rand(1) * (A_max - A_min) + A_min
    A_random = np.random.rand(3) * (color_max - color_min) + color_min
    A = A + A_random
    return np.clip(img * t + A * (1 - t), 0, 1)

def syn_data(hq_file, light_file, depth_file, rain_file, snow_file, out_file, 
             low, haze, rain, snow):
    file_list = os.listdir(hq_file)
    rain_list = os.listdir(rain_file)
    snow_list = os.listdir(snow_file)
    num_rain = random.sample(range(0,len(rain_list)),len(rain_list))
    num_snow = random.sample(range(0,len(snow_list)),len(snow_list))
    for i in range(1, len(file_list)):
        img = cv2.imread(hq_file+file_list[i])
        w, h, _ = img.shape
        light = cv2.cvtColor(cv2.imread(light_file + file_list[i]), cv2.COLOR_RGB2GRAY) / 255.0
        depth = cv2.imread(depth_file + file_list[i]) / 255.0
        rain_mask = cv2.imread(rain_file + rain_list[num_rain[i]]) / 255.0
        rain_mask = cv2.resize(rain_mask,(h,w))
        snow_mask = cv2.imread(snow_file + snow_list[num_snow[i]]) / 255.0
        snow_mask = cv2.resize(snow_mask, (h, w))
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)/ 255.0
        lq = img.copy()/255.0
        color_dis  = 1

        if low:
            lq = syn_low(lq, light, img_gray)
        if rain:
            lq = lq+rain_mask
        if snow:
            lq = lq*(1-snow_mask)+color_dis*snow_mask
        if haze:
            lq = syn_haze(lq, depth)

        # out = np.concatenate((lq*255.0,img),1)
        out = lq*255.0
        cv2.imwrite(out_file + file_list[i], out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
     # load model
    parser.add_argument("--hq-file", type=str, default = './data/clear/')
    parser.add_argument("--light-file", type=str, default = './data/light_map/')
    parser.add_argument("--depth-file", type=str, default = './data/depth_map/')
    parser.add_argument("--rain-file", type=str, default = './data/rain_mask/')
    parser.add_argument("--snow-file", type=str, default = './data/snow_mask/')
    parser.add_argument("--out-file", type=str, default = './out/')
    parser.add_argument("--low", action='store_true')
    parser.add_argument("--haze", action='store_true')
    parser.add_argument("--rain", action='store_true')
    parser.add_argument("--snow", action='store_true')
    
    args = parser.parse_args()

    syn_data(args.hq_file, args.light_file, args.depth_file, args.rain_file, 
             args.snow_file, args.out_file, args.low, args.haze, args.rain, args.snow)