import os
import pytesseract
from PIL import Image
import urllib
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.externals import joblib

if __name__ == '__main__':
    # 替换识别错误的数组
    replaceArray = {'|': 'j', '>': 'x', '<': 'x'}


    # 下载验证码图片
    def saveSimpleImg(index):
        if not os.path.exists('source_img'):
            os.mkdir('source_img')

        for i in range(50):
            u = urllib.request.urlopen('https://www.chinalife.com.cn/casServer/web/user/login/image.jsp')
            data = u.read()
            imgPath = "source_img/%d.png" % (i + int(index))
            with open(imgPath, 'wb') as f:
                f.write(data)


    # 灰度和二值化
    def binarizing(img, threshold):
        """传入image对象进行灰度、二值处理"""
        img = img.convert("L")  # 转灰度
        pixdata = img.load()
        w, h = img.size
        # 遍历所有像素，大于阈值的为黑色
        for y in range(h):
            for x in range(w):
                if pixdata[x, y] < threshold:
                    pixdata[x, y] = 0
                else:
                    pixdata[x, y] = 255
        # img.save('temp/0.png')
        return img


    # 进行垂直投影，返回的坐标数组
    def vertical(img):
        pixdata = img.load()
        w, h = img.size
        height = img.size[1]
        ver_list = []
        # 开始投影
        for x in range(w):
            black = 0
            for y in range(h):
                if pixdata[x, y] == 0:
                    black += 1
            ver_list.append(black)
        # 判断边界
        l, r = 0, 0
        flag = False
        cuts = []
        for i, count in enumerate(ver_list):
            # 阈值这里为0
            if flag is False and count > 0:
                l = i
                flag = True
            if flag and count == 0:
                r = i - 1
                flag = False
                # 这里没有用水平投影，偷懒用的0和高度
                cuts.append((l - 2, 0, r + 3, height))
        return cuts


    # 切割原始图片的黑边
    def cutImg(img):
        region = img.crop((2, 2, 66, 21))
        # region.save("temp/cut_first.png")
        return region


    # 根据返回的坐标，切割图片
    def getSplitImg(b_img, v):
        imgs = []
        for i, n in enumerate(v, 1):
            temp = b_img.crop(n)  # 调用crop函数进行切割
            imgs.append(temp)
            # temp.save("temp/cut_%s.png" % i)
        return imgs


    # 将切割好的图片，调用tesseract进行识别，然后保存到识别的目录里
    def ocrImgAndSave(fileName, imgs):
        for i, cur_img in enumerate(imgs):
            # 设置tesseract的工作目录
            pytesseract.pytesseract.tesseract_cmd = 'E:/Program Files (x86)/Tesseract-OCR/tesseract'
            recNum = pytesseract.image_to_string(cur_img, config='-psm 7 outputbase letters')
            print(recNum)
            for r in replaceArray:
                recNum = recNum.replace(r, replaceArray[r]).lower()
            if (recNum.isalpha() and len(recNum) == 1):
                # recNum = pytesseract.image_to_string(cur_img, config='-psm 10 outputbase digits')
                recdString = fileName + "-" + str(i + 1) + ".png"
                path = 'temp/' + recNum + "/"
                if not os.path.exists(path):
                    os.mkdir(path)
                imgPath = path + recdString
                cur_img.save(imgPath)


    # 提取特征值
    def extractLetters(path):
        x = []
        y = []
        # 遍历文件夹 获取下面的目录
        for root, sub_dirs, files in os.walk(path):
            for dirs in sub_dirs:
                # 获得每个文件夹的图片
                for fileName in os.listdir(path + '/' + dirs):
                    print(fileName)
                    # 打开图片
                    x.append(getletter(path + '/' + dirs + '/' + fileName))
                    y.append(dirs)

        return x, y


    # 提取SVM用的特征值, 提取字母特征值
    def getletter(fn):
        fnimg = cv2.imread(fn)  # 读取图像
        img = cv2.resize(fnimg, (8, 8))  # 将图像大小调整为8*8
        alltz = []
        for now_h in range(0, 8):
            xtz = []
            for now_w in range(0, 8):
                b = img[now_h, now_w, 0]
                g = img[now_h, now_w, 1]
                r = img[now_h, now_w, 2]
                btz = 255 - b
                gtz = 255 - g
                rtz = 255 - r
                if btz > 0 or gtz > 0 or rtz > 0:
                    nowtz = 1
                else:
                    nowtz = 0
                xtz.append(nowtz)
            alltz += xtz
        return alltz


    # 传入测试图片，进行识别测试
    def ocrImg(fileName):
        clf = joblib.load('data/letter.pkl')
        p = Image.open('test_img/%s' % fileName)
        p = cutImg(p)
        b_img = binarizing(p, 170)
        v = vertical(b_img)
        imgs = getSplitImg(b_img, v)
        captcha = []
        for i, img in enumerate(imgs):
            path = 'test_img/letter_%s.png' % i
            img.save(path)
            data = getletter(path)
            data = np.array([data])
            # print(data)
            oneLetter = clf.predict(data)[0]
            # print(oneLetter)
            captcha.append(oneLetter)
        captcha = [str(i) for i in captcha]
        print("the captcha is :%s" % ("".join(captcha)))


    # 获取分割图片的主方法
    def splitImgMain():
        for root, sub_dirs, files in os.walk('source_img'):
            for file in files:
                print('发现图片:' + file)
                p = Image.open('source_img/%s' % file)
                p = cutImg(p)
                b_img = binarizing(p, 170)
                v = vertical(b_img)
                # print(v)
                imgs = getSplitImg(b_img, v)
                ocrImgAndSave(file, imgs)


    # 进行向量机的训练SVM
    def trainSVM():
        array = extractLetters('temp')
        # 使用向量机SVM进行机器学习
        letterSVM = SVC(kernel="linear", C=1).fit(array[0], array[1])
        # 生成训练结果
        joblib.dump(letterSVM, 'data/letter.pkl')


    # # 1、先下载图片 指定开始顺序 默认0
    # saveSimpleImg(0)
    #
    # # 2、基于下载好的图片进行切割
    splitImgMain()
    #
    # # 3、基于准备好的图片，进行特征值的提取
    # trainSVM()

    # 4、找个图片试试，识别结果
    # ocrImg('test.png')
