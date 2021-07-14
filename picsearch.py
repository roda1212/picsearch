import argparse
import os
import glob
import lib.Log as Log
import cv2

def InitArgParser() -> argparse.ArgumentParser:
    """
    引数の初期化
    """
    parser = argparse.ArgumentParser(description='OfficeファイルのGrep検索')
    parser.add_argument('src', type=str, help='検索する画像')
    parser.add_argument('target', type=str, help='対象ディレクトリ')
    parser.add_argument('-out', type=str, help='結果CSVファイル', default='pic_result.csv')
    return parser

def EnumPictureFiles(target):
    jpg = glob.glob(f'{target}/**/*.jpg', recursive=True)
    jpeg = glob.glob(f'{target}/**/*.jpeg', recursive=True)
    png = glob.glob(f'{target}/**/*.png', recursive=True)
    bmp = glob.glob(f'{target}/**/*.bmp', recursive=True)
    return sorted(jpg + jpeg + png + bmp)

def Search(src, target):
    files = EnumPictureFiles(target)

    akaze = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    template = cv2.imread(src)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    kp1, des1 = akaze.detectAndCompute(template, None)
    # img1_akaze = cv2.drawKeypoints(template_gray, kp1, None, flags=4)
    # cv2.imwrite('sample.png', img1_akaze)

    i = 0
    for file in files:
        i += 1
        pic = cv2.imread(file)
        kp2, des2 = akaze.detectAndCompute(pic, None)

        matches = bf.match(des1, des2)
        matches = sorted(matches, key = lambda x:x.distance)
        print(file)
        print(len(matches))
        # for m in matches:
        #     print(m.distance)
        img3 = cv2.drawMatches(template, kp1, pic, kp2, matches[:50], None, flags=2)
        cv2.imwrite(f'sample_{i}.png', img3)
# https://emotionexplorer.blog.fc2.com/blog-entry-28.html
def Main():
    args = InitArgParser().parse_args()

    if not os.path.exists(args.src):
        Log.Error(f"検索画像が見つかりません（{args.src}）")
        return

    if not os.path.exists(args.target):
        Log.Error(f"ディレクトリが見つかりません（{args.pdf_file}）")
        return

    Search(args.src, args.target)
    

if __name__ == '__main__':
    Main()