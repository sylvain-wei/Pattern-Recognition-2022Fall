import numpy as np
import cv2


def cv_show(label, img):
    cv2.imshow(label, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 拼接器
class Processor():

    #拼接函数
    # 由于是像素为单位，reprojThresh一般在1-10左右
    def process(self, images, r=0.75, reprojThresh=4.0):
        # 获取输入图片
        imgL, imgR = images
        # 检测左右两张图片的SIFT关键特征点，并计算特征描述子:(128*128*3通道)形状的特征集
        kpsR, kpsR_cords, featuresR = self.keyPoints_features(imgR)
        kpsL, kpsL_cords, featuresL = self.keyPoints_features(imgL)

        # 匹配两张图片的所有特征点，返回匹配结果
        rt = self.match_points(kpsR_cords, kpsL_cords, featuresR, featuresL, r, reprojThresh)

        # 如果返回结果为空，没有匹配成功的特征点，退出算法
        if rt is None:
            return None

        # 否则，提取匹配结果
        # PerTrans是3x3视角变换矩阵
        matches1to2, PerTrans = rt

        # 将右侧图片进行透视变换，stitch_result是变换后图片
        stitch_result = cv2.warpPerspective(imgR, PerTrans, (imgR.shape[1] + imgL.shape[1], imgR.shape[0]))
        for y in range(imgR.shape[1] + imgL.shape[1]):
            if stitch_result[0,y,0] != 0:
                left = y
                break
        right = imgL.shape[1]
        cv_show('Result_RightImg_PerspectiveTransformation', stitch_result)
        # 生成匹配图片
        stitch_result = self.optimize(imgL, stitch_result, left, right)
        match_result = self.draw_matches(imgR, imgL, kpsR, kpsL, matches1to2)
        # 返回结果
        return stitch_result, match_result



    def keyPoints_features(self, img):
        # SIFT描述子生成器
        sift = cv2.SIFT_create()

        # 获取SIFT得到的特征点、(128*128*3通道)形状的特征集
        key_points, features = sift.detectAndCompute(img, None)

        # 获取到特征点的坐标信息
        kps_cords = np.float32([k.pt for k in key_points])

        # 返回特征点集，特征点对应坐标集，及对应的描述特征
        return key_points, kps_cords, features

    def match_points(self, cords_A, cords_B, featuresA, featuresB, r, reprojThresh):
        # 实例化暴力匹配器
        matcher = cv2.BFMatcher()
  
        # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

        matches = []
        matches1to2 = []
        for m in rawMatches:
            # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
            if len(m) == 2 and m[0].distance < m[1].distance * r:
            # 存储两个点在featuresA, featuresB中的索引值
                matches.append((m[0].trainIdx, m[0].queryIdx))
                matches1to2.append([m[0]])

        # 当筛选后的匹配对大于4时，计算视角变换矩阵
        if len(matches) > 4:
            # 获取匹配对的点坐标
            ptsA = np.float32([cords_A[i] for (_, i) in matches])
            ptsB = np.float32([cords_B[i] for (i, _) in matches])

            # 计算视角变换矩阵
            # 输入ptsA是源数据点二维坐标，ptsB是目标数据点二维坐标
            # 由于采用RANSAC需要指定reprojThresh这个阈值，其含义是 将点对视为内点的最大允许重投影错误阈值
            PerTrans, _ = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            # 返回结果
            return matches1to2, PerTrans

        # 如果匹配对小于4时，返回None
        return None

    def draw_matches(self, imageA, imageB, kpsA, kpsB, matches1to2):
        vis = cv2.drawMatchesKnn(imageA, kpsA, imageB, kpsB, matches1to2[:min(len(matches1to2), 40)], None, flags=2)
        return vis

    def optimize(self, imgL, stitch_result, left, right):
        """
           left: 重合部分左边界
           right: 重合部分右边界
        - return: 完美拼接后的图 
        """
        ovlp_width = right-left
        stitch_result[0:imgL.shape[0], 0:left] = imgL[:, 0:left]
        for y in range(left, right):
            # 由距离计算对应的权重
            alpha = (ovlp_width - (y - left)) / ovlp_width
            # 进行重叠部分的左图和右图相同位置的不同像素值的加权
            stitch_result[:, y, :] = alpha * imgL[:, y, :] + (1-alpha) * stitch_result[:, y, :]
        return stitch_result

