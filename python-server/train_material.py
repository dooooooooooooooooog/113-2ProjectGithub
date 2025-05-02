import glob
import numpy as np
from skimage import io, color
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

# 1. 擷取 LBP 特徵
def extract_lbp_hist(img_path, P=8, R=1):
    img = io.imread(img_path)
    gray = color.rgb2gray(img)
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, P+3),
                             range=(0, P+2))
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

# 2. 載入資料集：dataset/fleece/*.jpg、dataset/leather/*.jpg、dataset/other/*.jpg
X, y = [], []
for cls, code in [("fleece",201),("leather",202),("other",203)]:
    for path in glob.glob(f"dataset/{cls}/*.jpg"):
        X.append(extract_lbp_hist(path))
        y.append(code)
X = np.array(X); y = np.array(y)

# 3. 切訓練/測試集並訓練 SVM
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
clf = SVC(kernel="rbf", probability=True).fit(Xtr, ytr)
print("測試正確率：", clf.score(Xte, yte))

# 4. 儲存模型
with open("material_model.pkl","wb") as f:
    pickle.dump(clf, f)
