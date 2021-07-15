import argparse
import os
import glob
import lib.Log as Log
import cv2
from pathlib import Path
from jinja2 import Template, Environment, FileSystemLoader
import shutil

def InitArgParser() -> argparse.ArgumentParser:
    """
    引数の初期化
    """
    parser = argparse.ArgumentParser(description='OfficeファイルのGrep検索')
    parser.add_argument('src', type=str, help='検索する画像')
    parser.add_argument('target', type=str, help='対象ディレクトリ')
    return parser

def EnumPictureFiles(target):
    jpg = glob.glob(f'{target}/**/*.jpg', recursive=True)
    jpeg = glob.glob(f'{target}/**/*.jpeg', recursive=True)
    png = glob.glob(f'{target}/**/*.png', recursive=True)
    bmp = glob.glob(f'{target}/**/*.bmp', recursive=True)
    return sorted(jpg + jpeg + png + bmp)

class Result:
    point = 0
    target_img_path = ''
    match_img_path = ''
    def __init__(self) -> None:
        pass
    def SetTarget(self, target_img_path):
        self.target_img_path = target_img_path
    def SetMatch(self, match_img_path):
        self.match_img_path = match_img_path
    def SetPoint(self, point):
        self.point = point


def Search(src, target):
    files = EnumPictureFiles(target)
    files.sort()

    akaze = cv2.AKAZE_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    template = cv2.imread(src)
    # template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    kp1, des1 = akaze.detectAndCompute(template, None)
    # img1_akaze = cv2.drawKeypoints(template_gray, kp1, None, flags=4)
    # cv2.imwrite('sample.png', img1_akaze)

    r_dir = 'result/match'
    os.makedirs(r_dir, exist_ok=True)

    results = []

    i = 0
    for file in files:
        result = Result()
        result.SetTarget(file)
        i += 1
        pic = cv2.imread(file)
        kp2, des2 = akaze.detectAndCompute(pic, None)

        matches = bf.match(des1, des2)
        matches = sorted(matches, key = lambda x:x.distance)
        print(file)
        filtered = list(filter(lambda x: x.distance < 100, matches))
        # print(f'{len(matches)} -> {len(filtered)}')

        # for m in matches:
        #     print(m.distance)
        result.SetPoint(len(filtered))

        if len(filtered) > 0:
            img3 = cv2.drawMatches(template, kp1, pic, kp2, filtered[:10], None, flags=2)
            img_path = f'{r_dir}/sample_{i}.png'
            cv2.imwrite(img_path, img3)
            result.SetMatch(img_path)

        results.append(result)
    return results

# https://emotionexplorer.blog.fc2.com/blog-entry-28.html

def OutputResults(src, target, results):

    result_path = Path('./result').resolve()

    copied_src_path = Path(f'./result/{os.path.basename(src)}')
    shutil.copyfile(src, copied_src_path)
    copied_src_path = str(copied_src_path.relative_to('./result'))

    rs = []
    for result in results:
        # 比較先の絶対・相対パスを作成
        target_abs = Path(target).resolve()
        target_img_path_abs = Path(result.target_img_path).resolve()
        target_rel = target_img_path_abs.relative_to(target_abs)
        
        # 紐づけ画像は相対パス
        match_img_path = Path(result.match_img_path)

        # DictのListにする
        rs.append({ 'target_name': target_rel,
                    'target_img_path': str(target_img_path_abs),
                    'match_img_path': str(match_img_path.relative_to('./result')) if result.match_img_path != '' else '',
                    'judgement': result.point > 0})

    # HTMLマッピング＆書き出し
    env = Environment(loader=FileSystemLoader('./template'))
    template = env.get_template('index.html')
    html = template.render({'src':src, 'target':target, 'src_path': copied_src_path, 'results': rs })
    with open('result/index.html', 'w', encoding='utf-8', newline='') as f:
        f.write(html)

def Main():
    args = InitArgParser().parse_args()

    if not os.path.exists(args.src):
        Log.Error(f"検索画像が見つかりません（{args.src}）")
        return

    if not os.path.exists(args.target):
        Log.Error(f"ディレクトリが見つかりません（{args.pdf_file}）")
        return

    if os.path.exists('./result'):
        shutil.rmtree('./result')
    os.makedirs('./result', exist_ok=True)

    results = Search(args.src, args.target)
    OutputResults(args.src, args.target, results)

if __name__ == '__main__':
    Main()