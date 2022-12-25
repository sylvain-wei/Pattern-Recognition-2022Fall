from utils import Processor
import cv2

# 验证尺度不变性
test_1 = ['test_2_1.jpg', 'test_2_2.jpg']

# 验证旋转不变性
test_2 = ['test_3_1.jpg', 'test_3_2.jpg']

# 验证亮度变化不变性
test_3 = ['test_4_1.jpg', 'test_4_2.jpg']
# 随便拍的两张图
test_4 = ['3.jpg', '4.jpg']
def main(test_set):

    imgL = cv2.imread(test_set[0])
    for img in test_set[1:]:
        imgR = cv2.imread(img)
        # 把图片拼接成全景图
        processor = Processor()
        (stitch_result, match_result) = processor.process([imgL, imgR])
        imgL = stitch_result

        # 显示所有图片
        # cv2.imshow("Image A", imageA)
        # cv2.imshow("Image B", imageB)
        cv2.imshow("Keypoint Matches", match_result)
        cv2.imshow("Stitching Result", stitch_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(test_set=test_3)