import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

# 加载数据集
data = pd.read_csv('mbti_1.csv')

# 数据预处理
def preprocess_text(text):
    # 小写化
    text = text.lower()
    # 去除URL
    text = re.sub(r'http\S+', '', text)
    # 去除标点符号和数字
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# 应用预处理函数
data['posts'] = data['posts'].apply(preprocess_text)

# 标签编码
label_encoder = LabelEncoder()
data['type'] = label_encoder.fit_transform(data['type'])

# 划分数据集
X = data['posts']
y = data['type']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用TF-IDF进行文本向量化
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 使用支持向量机（SVM）进行分类
svm_model = SVC(kernel='linear', random_state=42, class_weight='balanced')
svm_model.fit(X_train_tfidf, y_train)
y_pred = svm_model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

# 随机森林
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_model.fit(X_train_tfidf, y_train)
y_pred_rf = rf_model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_, zero_division=0))

# 逻辑回归
lr_model = LogisticRegression(random_state=42, class_weight='balanced')
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred_lr, target_names=label_encoder.classes_, zero_division=0))

# 使用决策树算法进行分类
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_tfidf, y_train)
y_pred = dt_model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

# 使用K近邻（KNN）进行分类
knn_model = KNeighborsClassifier(n_neighbors=5)  # K近邻分类器，k=5
knn_model.fit(X_train_tfidf, y_train)
y_pred_knn = knn_model.predict(X_test_tfidf)
print("KNN Classifier Evaluation:")
print(classification_report(y_test, y_pred_knn, target_names=label_encoder.classes_, zero_division=0))

# 使用朴素贝叶斯（Naive Bayes）进行分类
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)
print("Naive Bayes Classifier Evaluation:")
print(classification_report(y_test, y_pred_nb, target_names=label_encoder.classes_, zero_division=0))
