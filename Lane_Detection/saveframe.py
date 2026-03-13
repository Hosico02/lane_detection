import os
import cv2

cap = cv2.VideoCapture('../example.mp4')  # xx.mp4为文件名

a = os.getcwd() + '/example'
os.mkdir(a)

frames_total = cap.get(7)
print('frames_total', frames_total)
frames_need = 50  # 生成的图片数，需小于总数否则报错
print('frames_need', frames_need)
n = 1
timeF = frames_total // frames_need  # 每过多少帧截取一张图片
print('timeF', timeF)
i = 0
while cap.isOpened():
    ret, frame = cap.read()
    if n % timeF == 0:
        i += 1
        print(i)
        cv2.imwrite('./图片/example{}.jpg'.format(i), frame)  # 生成后的放置路径
        if i == frames_need:
            break

    n = n + 1
    cv2.waitKey(1)

cap.release()


def jons_to_png():
    json_folder = os.getcwd()
    # 获取文件夹内的文件名
    FileNameList = os.listdir(json_folder)
    # 激活labelme环境
    os.system("conda activate labelme")
    for i in range(len(FileNameList)):
        # 判断当前文件是否为json文件
        if os.path.splitext(FileNameList[i])[1] == ".json":
            json_file = json_folder + "\\" + FileNameList[i]
            # 将该json文件转为png
            os.system("labelme_json_to_dataset " + json_file)


def ListFilesToTxt(dir, file, wildcard, recursion):
    exts = wildcard.split(" ")
    files = os.listdir(dir)
    for name in files:
        fullname = os.path.join(dir, name)
        if os.path.isdir(fullname) & recursion:
            ListFilesToTxt(fullname, file, wildcard, recursion)
        else:
            for ext in exts:
                if (name.endswith(ext)):
                    file.write("/CUlane/" + name + "\n")  # 存放合集（原图和json等）文件夹的名字
                    break


def Test():
    # 存放原图文件夹(只存放原图)路径
    dir = "/Users/mack/Desktop/work/车道线检测/原图"
    outfile = "trainlist.txt"  # 写入的txt文件名
    wildcard = ".jpg"  # 要读取的文件类型；
    file = open(outfile, "w")
    if not file:
        print("cannot open the file %s for writing" % outfile)
    ListFilesToTxt(dir, file, wildcard, 1)
    file.close()
