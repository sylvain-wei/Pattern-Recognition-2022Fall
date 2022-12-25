import numpy as np 
import cv2
import matplotlib.pyplot as plt

class SIFT():
    def __init__(self) -> None:
        self.sigma_0 = np.sqrt(1.6**2 - 0.5**2)
        self.k = 2 ** (1/3)
        self.N =  3 # sift paper's advised layers per octave
        self.O = None
        self.k_size=  3
        self.MAX_INTERP_TIMES = 5


    def detectAndCompute(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.O = int(np.log2(min(img.shape[0], img.shape[1])) - 3)   # num of octaves
        
        # 构造高斯金字塔
        gaussian_space = []
        for o in range(self.O):
            if o != 0:
                curr_img = self.subsample(gaussian_space[-3])
            else:
                curr_img = img
            gaussian_space.append(curr_img)
            sigma = self.sigma_0
            for r in range(1, self.N+3):
                sigma = self.sigma_0 * (self.k **(r-1))
                kernel = self.gaussian_kernel(self.k_size, sigma)
                res = self.conv(kernel, curr_img)
                gaussian_space.append(res)
                # cv2.imshow('Gaussian Space octave{}, layer{}'.format(o, r), res)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

        # 构造DoG，高斯差分空间
        DOG = []
        for o in range(self.O):
            for r in range(self.N+2):
                DOG.append(gaussian_space[o*(self.N+3)+r+1] - gaussian_space[o*(self.N+3)+r])
                # 展示高斯差分空间
                # cv2.imshow('DoG octave{}, layer{}'.format(o, r), DOG[o*(self.N+2)+r])
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

        # TODO:检测DoG中的极值点
        potential_x = []
        potential_y = []
        potential_o = []    # 与下面potential_r都是指的DoG空间中的组数
        potential_r = []
        for o in range(self.O):
            for r in range(1, self.N+2 - 1):
                cur_layer = DOG[o*(self.N+2)+r]
                up_layer = DOG[o*(self.N+2)+r+1]
                dn_layer = DOG[o*(self.N+2)+r-1]
                size_x = cur_layer.shape[0]
                size_y = cur_layer.shape[1]
                for x in range(1, size_x-1):
                    for y in range(1, size_y-1):
                        val = cur_layer[x][y]
                        if (abs(val) > 0.5*0.04/self.N and
                            ((val > max(np.max(up_layer[x-1:x+1, y-1:y+1]),
                                      np.max(dn_layer[x-1:x+1, y-1:y+1]))
                            and val > max(cur_layer[x-1][y-1], cur_layer[x-1][y],
                            cur_layer[x-1][y+1],cur_layer[x][y-1],cur_layer[x][y+1],
                            cur_layer[x+1][y-1],cur_layer[x+1][y],cur_layer[x+1][y+1]))
                            or (val < min(np.min(up_layer[x-1:x+1, y-1:y+1]),
                                         np.min(dn_layer[x-1:x+1, y-1:y+1]))
                            and val < min(cur_layer[x-1][y-1], cur_layer[x-1][y],
                            cur_layer[x-1][y+1],cur_layer[x][y-1],cur_layer[x][y+1],
                            cur_layer[x+1][y-1],cur_layer[x+1][y],cur_layer[x+1][y+1])))):
                                potential_x.append(x)
                                potential_y.append(y)
                                potential_o.append(o)    # 与下面potential_r都是指的DoG空间中的组数
                                potential_r.append(r)
            
        # 子像素插值，用相邻像素的差值代替求导（有限差分求导方法）
        prec_x = []
        prec_y = []
        prec_o = []
        prec_r = []
        prec_sigma = []
        for i in range(len(potential_x)):
            x = potential_x[i]
            y = potential_y[i]
            o = potential_o[i]
            r = potential_r[i]
            sigma = self.getSigma(o, r)
            X = np.array([x, y, sigma])
            for time in range(self.MAX_INTERP_TIMES):
                # 首先获得各个导数
                dx = (DOG[o*(self.N+2)+r][x+1][y] - DOG[o*(self.N+2)+r][x-1][y]) / 2
                dy = (DOG[o*(self.N+2)+r][x][y+1] - DOG[o*(self.N+2)+r][x][y-1]) / 2
                dxx = (DOG[o*(self.N+2)+r][x+1][y] + DOG[o*(self.N+2)+r][x-1][y] - 2 * DOG[o*(self.N+2)+r][x][y]) / 1
                dyy = (DOG[o*(self.N+2)+r][x][y+1] + DOG[o*(self.N+2)+r][x][y-1] - 2 * DOG[o*(self.N+2)+r][x][y]) / 1
                dxy = (DOG[o*(self.N+2)+r][x+1][y+1] + DOG[o*(self.N+2)+r][x-1][y-1] - DOG[o*(self.N+2)+r][x+1][y-1] - DOG[o*(self.N+2)+r][x-1][y+1]) / 4
                dsigma = (DOG[(o)*(self.N+2)+r+1][x][y] - DOG[(o)*(self.N+2)+r-1][x][y]) / 2
                dxsigma = (DOG[(o)*(self.N+2)+r+1][x+1][y] + DOG[(o)*(self.N+2)+r-1][x-1][y] - DOG[(o)*(self.N+2)+r+1][x-1][y] - DOG[(o)*(self.N+2)+r-1][x+1][y]) / 4
                dysigma = (DOG[(o)*(self.N+2)+r+1][x][y+1] + DOG[(o)*(self.N+2)+r-1][x][y-1] - DOG[(o)*(self.N+2)+r+1][x][y-1] - DOG[(o)*(self.N+2)+r-1][x][y+1]) / 4
                dsigmasigma = (DOG[(o)*(self.N+2)+r+1][x][y] + DOG[(o)*(self.N+2)+r-1][x][y] - 2 * DOG[o*(self.N+2)+r][x][y]) / 1
                
                f_X = np.array([dx, dy, dsigma])
                H_3 = np.array([[dxx, dxy, dxsigma], [dxy, dyy, dysigma], [dxsigma, dysigma, dsigmasigma]])
                # det_H_3 = np.linalg.det(H_3)
                # H_3_rev = (1 / det_H_3) * np.array([[],[],[]])
                if np.linalg.det(H_3) == 0:
                    break
                H_3_rev = np.linalg.inv(H_3)

                # 求得精确点的偏移向量
                dX = - np.matmul(H_3_rev, f_X)
                # 如果sigma、x、y变化都小于0.5，则表示已经不需要再更新了。
                max_thing = max([abs(item) for item in dX])
                D_ = DOG[o*(self.N+2)+r][x][y] + 0.5 * np.matmul(f_X.T, dX)
                if max_thing < 0.5:
                    break
                elif max_thing > 256/3 :
                    print("^^^^^False: 变化超出范围了^^^^^")
                    print()
                    break
                else:
                    X += np.array([round(dX[0]), round(dX[1]), round(dX[2])])
            # 保存精确的点
            if time > 0 and D_ >= 0.03 and max_thing <= 256/3 and abs(X[0]) < len(DOG[o*(self.N+2)+r]) -1 and abs(X[1]) < len(DOG[o*(self.N+2)+r][int(X[0])]) - 1 and abs(X[0]) > 0 and abs(X[1]) > 0:    # 要求至少进行了一次迭代且去掉对比度低的点
                prec_x.append(int(X[0]))
                prec_y.append(int(X[1]))
                prec_sigma.append(X[2])
                prec_o.append(o)
                prec_r.append(r)

        kp_x = []
        kp_y = []
        kp_sigma = []
        kp_o = []
        kp_r = []
        T_GAMMA = 10
        Treshold = (T_GAMMA + 1)**2 / T_GAMMA
        # 消除边缘响应
        for i in range(len(prec_x)):
            x = prec_x[i]
            y = prec_y[i]
            o = prec_o[i]
            r = prec_r[i]
            sigma = prec_sigma[i]

            dxx = (DOG[o*(self.N+2)+r][x+1][y] + DOG[o*(self.N+2)+r][x-1][y] - 2 * DOG[o*(self.N+2)+r][x][y]) / 1
            dyy = (DOG[o*(self.N+2)+r][x][y+1] + DOG[o*(self.N+2)+r][x][y-1] - 2 * DOG[o*(self.N+2)+r][x][y]) / 1
            dxy = (DOG[o*(self.N+2)+r][x+1][y+1] + DOG[o*(self.N+2)+r][x-1][y-1] - DOG[o*(self.N+2)+r][x+1][y-1] - DOG[o*(self.N+2)+r][x-1][y+1]) / 4
            
            Hessian = np.array([[dxx, dxy], [dxy, dyy]])
            
            # 矩阵的迹、行列式
            Tr_H = np.trace(Hessian)
            Det_H = np.linalg.det(Hessian)
            if (Tr_H**2 / Det_H) <= Treshold:
                # 不剔除
                kp_x.append(x)
                kp_y.append(y)
                kp_o.append(o)
                kp_r.append(r)
                kp_sigma.append(sigma)
        
        # 求取特征点的主方向
        main_angle = []
        for i in range(len(kp_x)):
            # 对于每一个特征点，先找到它的邻域
            x = kp_x[i]
            y = kp_y[i]
            o = kp_o[i]
            r = kp_r[i]
            sigma = kp_sigma[i]
            radius = 1.5 * sigma
            # 初始化统计量
            m = {}
            angles = [10 * i for i in range(36)]
            for angle in angles:
                m[angle] = 0

            for d_x in range(-round(radius), round(radius)+1):
                for d_y in range(-round(radius), round(radius)+1):
                    if np.sqrt(d_x**2 + d_y**2) <= radius:
                        x_ = x + d_x
                        y_ = y + d_y
                        if x_ > 0 and x_ < len(gaussian_space[o*(self.N+3)+r])-1:
                            if y_ > 0 and y_ < len(gaussian_space[o*(self.N+3)+r][x_]) - 1:
                                m_ = np.sqrt((gaussian_space[o*(self.N+3)+r][x_+1][y_] - 
                                              gaussian_space[o*(self.N+3)+r][x_-1][y_])**2
                                            +(gaussian_space[o*(self.N+3)+r][x_][y_+1] -
                                              gaussian_space[o*(self.N+3)+r][x_][y_-1])**2)
                                if (gaussian_space[o*(self.N+3)+r][x_+1][y_] - 
                                              gaussian_space[o*(self.N+3)+r][x_-1][y_]) != 0:
                                    theta = np.arctan((gaussian_space[o*(self.N+3)+r][x_][y_+1] -
                                                gaussian_space[o*(self.N+3)+r][x_][y_-1])
                                                /(gaussian_space[o*(self.N+3)+r][x_+1][y_] - 
                                                gaussian_space[o*(self.N+3)+r][x_-1][y_]))
                                else:
                                    continue
                                theta = round(theta * 180 / np.pi)
                                if theta < 0:
                                    theta += 360
                                if theta % 10 > 5:
                                    m[int((((theta//10)+1)*10) % 360)] += m_
                                else:
                                    m[int(((theta//10)*10) % 360)] += m_
            max_angle, max_m = 0, 0
            for i in range(36):
                if m[10*i] > max_m:
                    max_m = m[10*i]
                    max_angle = 10*i
            # 找到主方向
            main_angle.append(max_angle)
            # 展示当前的主方向
            line_len = 10
            x__ = int(round(x + line_len * np.cos(max_angle * np.pi / 180)))
            y__ = int(round(y + line_len * np.sin(max_angle * np.pi / 180)))
            if o <= 2:         
                continue
            if x__ >= 0 and x__ < len(gaussian_space[o*(self.N+3)+r]) and y__ >= 0 and y__ < len(gaussian_space[o*(self.N+3)+r][x__]):
                img_ = cv2.arrowedLine(gaussian_space[o*(self.N+3)+r], (x, y), (x__, y__), (0,0,255), 2, 8, 0, 0.3)
                cv2.imshow('Main Direction: octave{}, layer{}'.format(o, r), img_)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
        # new_G = {}

        # for i in range()






        # for i in range(len(kp_x)):
        #     # 对于每一个特征点，先找到它的邻域
        #     x = kp_x[i]
        #     y = kp_y[i]
        #     o = kp_o[i]
        #     r = kp_r[i]
        #     sigma = kp_sigma[i]
        #     radius = (np.sqrt(2) * 3 * sigma * (4 + 1))
        #     for d_x in range(-round(radius), round(radius)+1):
        #         for d_y in range(-round(radius), round(radius)+1):
        #             if np.sqrt(d_x**2 + d_y**2) <= radius:
        #                 x_ = x + d_x
        #                 y_ = y + d_y
        #                 if x_ > 0 and x_ < len(gaussian_space[o*(self.N+3)+r])-1:
        #                     if y_ > 0 and y_ < len(gaussian_space[o*(self.N+3)+r][x_]) - 1:
        #                         Matrix = np.array([[np.cos(main_angle[i]), -np.sin(main_angle[i])],[np.sin(main_angle[i]), np.cos(main_angle[i])]])
        #                         X = np.array([x+d_x, y+d_y])
        #                         X_ = np.matmul(Matrix, X)
        #                         X_ = np.array([round(X_[0]), round(X_[1])])
        #                         gaussian_space[X_] = gaussian_space[o*(self.N+3)+r][x][y]




    def subsample(self, img_3):
        """
            删除所有偶数行和列
        """
        return img_3[::2, ::2]

    def getSigma(self, o, r):
        return self.sigma_0 * 2 ** (o + r / self.N)
    
    def gaussian_kernel(self, size, sigma):
        """
        - params:
            - size: kernel's size (squared)
            - sigma
        - return:
            - kernel 2D-array like
        """
        max_interval = int((size - 1) / 2)# for size=5, [-2, -1, 0, 1, 2]
        kernel = np.zeros((size, size))
        for x in range(-max_interval, max_interval + 1):
            for y in range(-max_interval, max_interval + 1):
                kernel[x+max_interval][y+max_interval] = ((1 / (2 * np.pi  
                    * (sigma ** 2))) * np.exp(-(x**2 + y**2) / 2*(sigma**2)))
        sum = np.sum(kernel)
        kernel = kernel / sum

        return kernel

    def expand_img(self, img, k_size, img_size):
        """
        expand img in order that after convolution, new image has the same 
            shape like original
        """
        img_size_x, img_size_y = img_size[0], img_size[1]
        expand_img = np.zeros((img_size_x + k_size - 1, img_size_y + k_size - 1))
        expand_img[int((k_size-1)/2): int((k_size-1)/2+img_size_x), int((k_size-1)/2): int((k_size-1)/2+img_size_y)] = img
        for y in range(int((k_size-1)/2)):
            expand_img[:, y] = expand_img[:, int((k_size-1)/2 + ((k_size-1)/2 - y))]
            expand_img[:, -1-y] = expand_img[:, int(-1-((k_size-1)/2 + ((k_size-1)/2 - y)))]
        for x in range(int((k_size-1)/2)):
            expand_img[x, :] = expand_img[int((k_size-1)/2 + ((k_size-1)/2 - y)), :]
            expand_img[-1-x, :] = expand_img[int(-1-((k_size-1)/2 + ((k_size-1)/2 - x))), :]
        return expand_img

    def conv(self, kernel, img):
        """
            input should be just one channel
        """
        k_size = kernel.shape[0]
        img_size = img.shape
        expand_img = self.expand_img(img, k_size, img_size)
        new_img = np.zeros_like(img)

        for x in range(img_size[0]):
            for y in range(img_size[1]):
                new_img[x][y] = 0
                for i in range(k_size):
                    for j in range(k_size):
                        new_img[x][y] += expand_img[x+i][y+j] * kernel[i][j]
        new_img = new_img.astype(np.uint8)
        return new_img

  





if __name__ == '__main__':
    sift = SIFT()
    # kernel = sift.gaussian_kernel(3,0.5)
    img = cv2.imread('1.jpg')
    # cv2.imshow('aaa', sift.conv(kernel, img[:,:,0]))

    # res1 = np.expand_dims(sift.conv(kernel, img[:,:,0]), axis=2)
    # res2 = np.expand_dims(sift.conv(kernel, img[:,:,1]), axis=2)
    # res3 = np.expand_dims(sift.conv(kernel, img[:,:,2]), axis=2)
    # res = np.stack((res1, res2, res3), axis=-1).squeeze()
    # cv2.imshow('1', res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    sift.detectAndCompute(img)
