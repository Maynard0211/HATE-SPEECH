#hatespeech.py
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from sklearn.preprocessing  import LabelEncoder
import warnings 
warnings.filterwarnings('ignore')
import os
import plotly.graph_objs as go
import joblib
from os import path
import itertools
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from gensim.model import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud 
from keras.models import Sequential, load_mode
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout, Conv1D, MaxPooling1D
from tensorflow.python.keras import models, layers 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model 
from keras.callbacks import EarlyStopping 
from sklearn.metrics import roc_auc_score, roc_curve  
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold 
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier 
from xgboost import XGBClassifier
from sklearn.neutral_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, \
    LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score 
from sklearn.metrics import classification_report, f1_score, plot_confusion_matrix 
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE 
from sklearn.model_selection import learning_curve
from mlxtend.plotting import plot_decision_regions
 
# Đọc tập dữ liệu
curr_path = os.getcwd()
df = pd.read_csv(curr_path +"/train.csv")

# Kiểm tra bản in
print(df.shape)

# Đọc cột
print("Data Columns -->".list(df.columns))

# Trích xuất tweet và label
tweet = df['tweet'].tolist()
label = df['label'].tolist()

# In một số ví dụ về tweet và label
print("Tweet:")
for i in range(5):
    print("{} --- {}".format(tweet[i],label[i]))

# Kiểm tra giá trị null
print(df.isnull().sum())
print('Total number of null values: ', df.isnull().sum().sum())

# Xóa cột "id"
df = df.drop("id", axit = 1)

# Kiểm tra thông tin tệp dữ liệu
print(df.info())

# Định nghĩa chức năng tạo biểu đồ tròn và biểu đồ thanh với subpots
def plot_piechar(df,var, title=''):
    plt.figure(figsize=(25,10))
    plt.subplot(121)
    label_list = list(df[var].value_count().index)
    
    df[var].value_counts().plot.pie(autopct = "%1.1f%%", \
        colors = sns.color_palette("prism",7), \
        startangle = 60, labels= label_list, \
        wedgeprops = {"linewidth":2,"edgecolor": "k"}, \
        shadow = True, textprops={'fontsize':20}  
    )
    plt.title("Distribution of"+ var + "variable"+title, fontsize = 25)
    
    plt.subplot(122)
    ax = df[var].value_count().plot(kind="barh")

    for i,j in enumerate(df[var.value_count().values]):
        ax.text(.7,i,j,weight = "bold", fontsize= 20)
    plt.title("Count of"+ var + "cases"+ title, fontsize = 25)
    plt.show()
plot_piechar(df,'label')

# Phân phối độ dài của các tính năng tweet
fig, ax1 = plt.subplots(figsize=(35,20))
plt.subplots_adjust(wspace=0.25, hspace =0.25)

sns.histplot(data = df.tweet.str.len(),\
    ax=ax1, kde = True, bins=100, line_kws={'lw':5})
ax1.set_title('Histogram Distribution of Length of Tweet', \
    fontsize=35)
ax1.set_xlabel('Length of Tweet', fontsize=35)
ax1.set_ylabel('Count', fontsize=35)
for p in ax1.patches:
    ax1.annotate(format(p.get_height(), '.0f'), \
        (p.get_x() + p.get_wigth() / 2., p.get_height()), \
        ha = 'center', va = 'center',xytext = (0, 10), \
        weight = "bold", fontsize= 30, \
        textcoords = 'offser points')
    
# Phân loại chiều dài của tweet
df['len_tweet'] = df.tweet.str.len()
labels = ['0-50' , '50-75' , '75-100' , '100-125' , '125-150' ,'150-500']
df['len_tweet'] = pd.cut(df['len_tweet'], \
    [0, 50, 75, 100, 125, 150, 500], labels=labels)
plot_piechar(df,'len_tweet')

# biểu đồ phân phối độ dài của tweet không có tweet phân biệt chủng tộc/phân biệt giới tính
# trong tập dữ liệu ở biểu đồ hình tròn và biểu đồ thanh
plot_piechar(df[df.label==1], "len_tweet", \
    "with racits/sexist tweets")

# Đặt nhãn bên trong thanh xếp chồng lên nhau
def put_label_stacked_bar(ax, fontsize):
    
    #  Vá lỗi là mọi thứ bên trong 'char'
    for rect in ax.patches:
        
        # Tìm và xác định vị trí của mọi thứ
        height = rect.get_height()
        width = rect.get_width()
        x = rect.get_x()
        y = rect.get_y()
        
        # Chiều cao của thanh là giá trị dữ liệu và có thể được sử dụng làm nhãn
        label_text = f'{height:.0f}'
        
        # ax.text(x, y, text)
        label_x = x + width / 2
        label_y = y + height / 2
        
        # Vẽ duy nhất một đồ thị khi chiều cao lớn hơn giá trị đã chỉ định
        if height > 0:
            ax.text(label_x, label_y, label_text, \
            ha = 'center', va = 'center', \
            weight = "bold", fontsize = fontsize)
            
# Vẽ đồ thị một biến so với một biến khác
def dist_one_vs_another_plot(df, cat1,cat2, title=""):
    fig = plt.figue(figsize=(18,12))
    cmap = plt.cm.Blues 
    cmap1 = plt.cm.coolwarm_r 
    ax1 = fig.add_subplot(111)
    
    group_by_stat = df.proupby([cat1, cat2]).size() 
    group_by_stat.unstack().plot(kind='bar', \
        stacked=True, ax = ax1, grid = True) 
    ax1.set_title('Stacked Bar Plot of' + \
        cat1 + '(number of cases)' + title, fontsize = 30)
    ax1.set_ylabel('Number of Cases', fontsize=20)
    ax1.set_xlabel(cat1, fontsize=20)
    put_label_stacked_bar(ax1,17)
    plt.show()
    
# Vẽ biểu đồ phân phối độ dài của tweet theo nhãn trong biểu đồ thanh xếp chồng lên nhau
dist_one_vs_another_plot(df,'len_tweet', 'label')

################################# TEXT PROCESSING #############################
racist = df[df['label']==1].tweet 

# Vẽ đồ thị đám mây cho phân biệt chủng tộc/phân biệt giới tính trên thế giới
plt.figue(figsize=(24,20))
world_cloud_racist=WordCloud(min_font_size=3, max_words=3200,width=1600,height =720).generate("".join(racist))
plt.imshow(world_cloud_racist, interpolation='bilinear')
plt.grid(False)

# Xóa các từ dừng
def remove_stopwords(text):
    text = ''.join([word for word in text.split() if word not in (stopwords.words('english'))])
    return text

# xóa các url
def remove_url(text):
    url = re.compile(r'http?://\S+|www\. \S+')
    return url.sub(r'', text)

# Xóa dấu câu
def remove_punct(text):
    table = str.maketrans('','', string.punctuation)
    return text.traslate(table)

# Xóa link html
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r',text')

# Xóa  @username(tên người dùng)
def remove_username(text):
    return re.sub('@[^\s]+','',text)

# Xóa emojis(biểu tượng cảm xúc)
def remove_emoji(text):
    emoji_pattern = re.compile("["
                    u"\U0001F600-\U0001F64F" # emoticons
                    u"\U0001F300-\U0001F5FF" # symbols & pictographs
                    u"\U0001F680-\U0001F6FF" # transport & map symbols
                    u"\U0001F1E0-\U0001F1FF" # flags (iOs)
                    u"\U00002702-\U000027B0"
                    u"\U000024C2-\U0001F251"
                    "]+", flags = re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Rút gọn văn bản
def decontraction(text):
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"won\'t've", "will not have", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"don\'t", "do not", text)
    text = re.sub(r"can\'t've", "can not have", text)
    text = re.sub(r"ma\'am", "madam", text)
    text = re.sub(r"let\'s", "let us", text)
    text = re.sub(r"ain\'t", "am not", text)
    text = re.sub(r"shan\'t", "shall not", text)
    text = re.sub(r"sha\n't", "shall not", text)
    text = re.sub(r"o\'clock", "off the clock", text)
    text = re.sub(r"y\'all", "you all", text)
    text = re.sub(r"n\'t", "not", text)
    text = re.sub(r"n\'t've", "not have", text)
    text = re.sub(r"\'re", "are", text)
    text = re.sub(r"\'s", "is", text)
    text = re.sub(r"\'d", "would", text)
    text = re.sub(r"\'d've", "would have", text)
    text = re.sub(r"\'ll", "will", text)
    text = re.sub(r"\'ll've", "will have", text)
    text = re.sub(r"\'ve", "have", text)
    text = re.sub(r"\'m", "am", text)
    return text

# Phân tách chữ và số
def seperate_alphanumeric(text):
    words = text
    words = re.findall(r"[^\W\d_]+|\d+", words)
    return " ".join(words)
def cont_rep_char(text):
    tchr = text.group(0)
    if len(tchr) > 1:
        return tchr[0:2]
def unique_char(rep, text):
    substitute = re.sub(r'(\w)\1+', rep, text)
    return substitute
def char(text):
    substitute = re.sub(r'[^a-zA-Z0-9]','', text)
    return substitute
df['final_tweet'] = df['tweet'].copy(deep=True)

def clean_tweet(df):
    
# Áp dụng chức năng trên Văn bản
    df['final_tweet'] = df['final_tweet'].apply(lambda x : remove_username(x))
    df['final_tweet'] = df['final_tweet'].apply(lambda x : remove_url(x))
    df['final_tweet'] = df['final_tweet'].apply(lambda x : remove_emoji(x))
    df['final_tweet'] = df['final_tweet'].apply(lambda x : decontraction(x))
    df['final_tweet'] = df['final_tweet'].apply(lambda x : seperate_alphanumeric(x))
    df['final_tweet'] = df['final_tweet'].apply(lambda x : unique_char(cont_rep_char, x))
    df['final_tweet'] = df['final_tweet'].apply(lambda x : char(x))
    df['final_tweet'] = df['final_tweet'].apply(lambda x : x.lower())
    df['final_tweet'] = df['final_tweet'].apply(lambda x : remove_stopwords(x))
    
# Làm sạch các tweet
clean_tweet(df)

# In kết quả
print(df['final_tweet'])
print(df['tweet'])

unique_chars = pd.Series([char for sentence in df["final_tweet"] for char in sentence]).unique()
print("Number of unique chars:", len(unique_chars))
print(unique_chars)

# Save final tweet
joblib.dump(df['final_tweet'], 'final_tweet.pkl')

################################ MACHINE LEARNING ####################################

# Trích xuất các biến đầu ra và đầu vào
if path.isfile('final_tweet.pkl'):
    
    # Load final text
    X = joblib.load('final_tweet.pkl')
else:
    clean_tweet(df)

    # Save final text 
    joblib.dump(df['final_tweet'], 'final_tweet.pkl')
    
    X = df['final_tweet']
y = df["label"]

# Save X and y
joblib.dump(X, 'X.pkl')
joblib.dump(y, 'y.pkl')

print(X.shape) 
print(y.shape) 
'''
X=X[0:5000]
y=y[0:5000]
''' 
def apply_tfidfvectorizer(X, y):
    
    # Áp dụng TFIDF
    tfidf = TfidfVectorizer()
    Xtfidf_final = tfidf.fit_transform(X)
    print(Xtfidf_final.shape)

    # Lấy mẫu lại dữ liệu bằng SMOTE
    sm = SMOTE(random_state=42)
    Xtfidf_sm, ytfidf_sm = sm.fit_resample(Xtfidf_final, y)
    print( Xtfidf_sm.shape) 
    print( ytfidf_sm.shape) 
    
    # Chia nhỏ dữ liệu để được đào tạo và kiểm tra TFIDF
    Xtfidf_train, Xtfidf_test, ytfidf_train, ytfidf_test = train_test_split(Xtfidf_sm,
    ytfidf_sm, shuffle=True, test_size = 0.2, random_state = 2023, stratify = ytfidf_sm)
    Xtfidf_train = Xtfidf_train.astype('float32')
    Xtfidf_test = Xtfidf_test.astype('float32')
    ytfidf_train = ytfidf_train.astype('float32')
    ytfidf_test = ytfidf_test.astype('float32')
    
    # Lưu vào tệp pkl
    joblib.dump(Xtfidf_train, 'Xtfidf_train.pkl')
    joblib.dump(Xtfidf_test, 'Xtfidf_test.pkl')
    joblib.dump(ytfidf_train, 'ytfidf_train.pkl')
    joblib.dump(ytfidf_test, 'ytfidf_test.pkl')
def apply_countvectorizer(X,y):
    
    # Áp dụng Count Vectorizer
    countVect = CountVectorizer()
    XCV_final = countVect.fit_transform(X)
    
    # Lấy mẫu lại dữ liệu bằng SMOTE
    sm = SMOTE(random_state=42)
    XCV_sm, yCV_sm = sm.fit_resample(XCV_final, y)
    print(XCV_final.shape)
    
    # Chia nhỏ dữ liệu để đào tạo và thử nghiệm bằng Count Vectorizer
    XCV_train, XCV_test, yCV_train, yCV_test = train_test_split(XCV_sm, yCV_sm, test_size=
 0.2, shuffle=True, random_state = 2023, stratify=yCV_sm)
    XCV_train = XCV_train.astype('float32')
    XCV_test = XCV_test.astype('float32')
    yCV_train = yCV_train.astype('float32')
    yCV_test = yCV_test.astype('float32')
    
    # Lưu vào tệp pkl
    joblib.dump(XCV_train, 'XCV_train.pkl')
    joblib.dump(XCV_test, 'XCV_test.pkl')
    joblib.dump(yCV_train, 'yCV_train.pkl')
    joblib.dump(yCV_test, 'yCV_test.pkl')
def apply_hashingvectorizer(X,y):
    
    # Áp dụng đếm Vectorizer
    hashVect = HashingVectorizer()
    XHV_final = hashVect.fit_transform(X)
    
    # Lấy mẫu lại dữ liệu bằng SMOTE
    sm = SMOTE(random_state=42)
    XHV_sm, yHV_sm = sm.fit_resample(XHV_final, y)
    print(XHV_final.shape) 
    
    # Chia dữ liệu để đào tạo và thử nghiệm Count Vectorizer
    XHV_train, XHV_test, yHV_train, yHV_test = train_test_split(XHV_sm, yHV_sm, test_size=
 0.2, shuffle=True, random_state = 2023, stratify=yHV_sm)
    XHV_train = XHV_train.astype('float32')
    XHV_test = XHV_test.astype('float32')
    yHV_train = yHV_train.astype('float32')
    yHV_test = yHV_test.astype('float32')
    
    # Lưu vào tệp pkl 
    joblib.dump(XHV_train, 'XHV_train.pkl')
    joblib.dump(XHV_test, 'XHV_test.pkl')
    joblib.dump(yHV_train, 'yHV_train.pkl')
    joblib.dump(yHV_test, 'yHV_test.pkl')
def loadtfidf_files():
    Xtfidf_train = Xtfidf_train.astype('float32')
    Xtfidf_test = Xtfidf_test.astype('float32')
    ytfidf_train = ytfidf_train.astype('float32')
    ytfidf_test = ytfidf_test.astype('float32')
    return Xtfidf_train, Xtfidf_test, ytfidf_train, ytfidf_test
def loadcount_files():
    XCV_train = XCV_train.astype('float32')
    XCV_test = XCV_test.astype('float32')
    yCV_train = yCV_train.astype('float32')
    yCV_test = yCV_test.astype('float32')
    return XCV_train, XCV_test, yCV_train, yCV_test 
def loadhashing_files():
    XHV_train = XHV_train.astype('float32')
    XHV_test = XHV_test.astype('float32')
    yHV_train = yHV_train.astype('float32')
    yHV_test = yHV_test.astype('float32')
    return XHV_train, XHV_test, yHV_train, yHV_test 

if path.isfile('Xtfidf_train.pkl'):
    Xtfidf_train, Xtfidf_test, ytfidf_train, ytfidf_test = loadtfidf_files()
    XCV_train, XCV_test, yCV_train, yCV_test = loadcount_files()
    XHV_train, XHV_test, yHV_train, yHV_test = loadcount_files()
else: 
    apply_tfidfvectorizer(X,y)
    apply_countvectorizer(X,y)
    apply_hashingvectorizer(X,y)
    
    Xtfidf_train, Xtfidf_test, ytfidf_train, ytfidf_test = loadtfidf_files()
    XCV_train, XCV_test, yCV_train, yCV_test = loadcount_files()
    XHV_train, XHV_test, yHV_train, yHV_test = loadcount_files()

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=None,
train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")
    
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1) 
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    
    # Đồ thị đường cong học tập
    axes[0].grid()
    axes[0].fill_between(train_sizes, \
        train_scores_mean - train_scores_std,\
        train_scores_mean + train_scores_std,\
        alpha=0.1, color="r")
    axes[0].grid()
    axes[0].fill_between(train_sizes,\
        test_scores_mean - test_scores_std, \
        test_scores_mean + test_scores_std, \
        alpha=0.1, color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', \
        color="r", label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', \
        color="g", label="Cross-validation score")
    axes[0].legend(loc="best")
    
    # Đồ thị n_sample so với fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, \
        fit_times_mean - fit_times_std, \
        fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Scalability of the model")
    
    # Đồ thị fit_time so với score
    axes[2].grid()
    axes[2].plot(test_scores_mean, fit_times_mean, 'o-')
    axes[2].fill_between(fit_times_mean, \
        test_scores_mean - test_scores_std, \
        test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")
    
    return plt
 
def plot_real_pred_val(Y_test, ypred, name):
    plt.figure(figsize = (15,10))
    acc = accuracy_score(Y_test,ypred)
    plt.scatter(range(len(ypred)), ypred,color ="blue", \
        lw=5,label = "Predicted")
    plt.scatter(range(len(Y_test)), \
        Y_test, color="red", label="Actual")
    plt.title("Predicted Values vs True Values of" + name, \
        fontsize=30)
    plt.xlabel("Accuracy: " + str(round((acc*100),3)) + "%", fontsize=20)
    plt.ylabel("Response", fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.75, lw=1, ls='-.')
    plt.show()
    
def plot_cm(Y_test, ypred, name=''):
    plt.figure(figsize=(15,10))
    ax = plt.subplot()
    cm = confusion_matrix(Y_test,ypred)
    sns.heatmap(cm, annot=True, linewidth=0.7, \
        linecolor='red', fmt ='g', cmap ="YlOrBr")
    plt.title(name + 'Confusion Matrix')
    plt.xlabel('Y predict')
    plt.ylabel('Y test')
    ax.xaxis.set_ticklabels(['0', '1'])
    ax.yaxis.set_ticklabels(['0', '1'])
    plt.show()
    return cm
def train_model(model, X, y):
    model.fit(X, y)
    return model
def predict_model(model, X, proba=False):
    if  ~proba:
        y_pred = model.predict(X)
    else:
        y_pred_proba = model.predict_proba(X)
        y_pred = np.argmax(y_pred_proba, axis = 1)
    return y_pred
list_scores = []
def run_model(name, model, X_train, X_test, y_train, y_test, fc, proba=False):
    print(name)
    print(fc)
    
    model = train_model(model, X_train, y_train)
    y_pred = predict_model(model, X_test, proba)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average ='weighted')
    precision = precision_score(y_test, y_pred, average ='weighted')
    f1 = f1_score(y_test, y_pred, average = 'weighted')
    
    print('accuracy:', accuracy)
    print('recall:', recall)
    print('precision:', precision)
    print('f1:', f1)
    print(classification_report(y_test, y_pred))
    
    plot_cm(y_test, y_pred, name) 
    plot_real_pred_val(y_test, y_pred, name)
    plot_learning_curve(model, name, X_train, y_train, cv=3);
    plt.show()
    list_scores.append({'Model Name': name, 'Feature Scaling':fc, 'Accuracy': accuracy,
 'Recall': recall, 'Precision':precision, 'F1':f1})

feature_vectorization = {
    'Hashing Vectorizer': (XHV_train, XHV_test, yHV_train, yHV_test),
    'TFIDF': (Xtfidf_train, Xtfidf_test, ytfidf_train, ytfidf_test),
    'Count Vectorizer': (XCV_train, XCV_test, yCV_train, yCV_test),
} 

model_svc = SVC(random_state=2023, probalility=True)
for fc_name, value in feature_vectorization.items():
    X_train, X_test, y_train, y_test = value
    run_model('SVC', model_svc, X_train, X_test, y_train, y_test, fc_name)
logreg = LogisticRegression(solver='lbfgs', max_iter=5000, random_state=2023)
for fc_name, value in feature_vectorization.items():
    X_train, X_test, y_train, y_test = value
    run_model('Logistic Regression', logreg, X_train, \
        X_test, y_train, y_test, fc_name, proba=True)
for fc_name, value in feature_vectorization.items():
    scores_1 =[]
    X_train, X_test, y_train, y_test = value
    for i in range(2,10):
        knn = KNeighborsClassifier(n_neighbors= i)
        knn.fit(X_train, y_train)
        
        scores_1.append(accuracy_score(y_test,\
            knn.predict(X_test)))
    max_val = max(scores_1)
    max_index = np.argmax(scores_1) + 2
    
    knn = KNeighborsClassifier(n_neighbors = max_index)
    knn.fit(X_train, y_train)
    
    run_model(f'KNeighborsClassifier n_neighbors = {max_index}', \
        knn, X_train, X_test, y_train, y_test, \
        fc_name, proba=True)

for fc_name, value in feature_vectorization.items():
    X_train, X_test, y_train, y_test = value
    
    dt = DecisionTreeClassifier()
    parameters = {'max_depth':np.arange(1,21,1), \
        'randoom_state':[2023]}
    searcher = GridSearchCV(dt, parameters)
    
    run_model('Decision Tree Classifier', searcher, X_train, \
        X_test, y_train, y_test, fc_name, proba=True)
rf = RandomForestClassifier(n_estimators = 100, \
    max_depth = 20, random_state= 2023)

for fc_name, value in feature_vectorization.items():
    X_train, X_test, y_train, y_test = value
    run_model('Random Forest Classifier', rf, X_train, \
        X_test, y_train, y_test, fc_name, proba=True)
    
gbt = GradientBoostingClassifier(n_estimators = 50, max_depth=10, subsanple =0.8,
max_features=0.2, random_states=2023)
for fc_name, value in feature_vectorization.item():
    X_train, X_test, y_train, y_test = value
    run_model('Gradient Boosting Classifier', gbt, X_train, X_test, y_train, y_test,
fc_name, proba=True)   
    
for fc_name, value in feature_vectorization.item():
    X_train, X_test, y_train, y_test = value
    xgb = XGBClassifier(n_estimators = 50, max_depth = 10, random_state=2023,
use_label_encoder=False, eval_metric='mlogloss')
    run_model('XGBoost Classifier', xgb, X_train, X_test, y_train, y_test, fc_name, proba=True)
    
lgbm = LGBMClassifier(max_depth = 10, n_estimators=50, subsample=0.8, random_state = 2023)
for fc_name, value in feature_vectorization.item(): 
    X_train, X_test, y_train, y_test = value
    run_model('Lightgbm Classifier', lgbm, X_train, X_test, y_train, y_test,
fc_name, proba=True) 
    
ada = AdaBoostClassifier(n_estimators = 50, learning_rate=0.01)
for fc_name, value in feature_vectorization.item(): 
    X_train, X_test, y_train, y_test = value
    run_model('ADA', ada, X_train, X_test, y_train, y_test,fc_name, proba=True)

mlp = AdaBoostClassifier(random_state = 2023, max_iter=200)
for fc_name, value in feature_vectorization.item(): 
    X_train, X_test, y_train, y_test = value
    run_model('MLP', mlp, X_train, X_test, y_train, y_test,fc_name, proba=True)

############################ DEEP LEARNING ###################################
############################### LSTM #########################################
max_words = 5000
max_len = 1000

def tokenize_pad_sequences(text):
    '''
    Hàm này mã hóa văn bản đầu vào thành các chuỗi xen kẽ và sau đó
    đệm từng chuỗi có cùng độ dài
    '''
    # Mã thông báo dạng văn bản
    tokenizer = Tokenizer(num_words=max_words, lower=True, split='')
    tokenizer.fit_on_texts(text)
    
    # Chuyển đổi văn bản thành một chuỗi kí tự trung gian
    X = tokenizer.texts_to_sequences(text)
    
    # Các chuỗi Pad có cùng độ dài
    X = pad_sequences(X, padding='post', maxlen=max_len)
    
    # Trả về tuần tự (trình tự)
    return X, tokenizer

print('Before Tokenization & Padding \n', df['final_tweet'][0], '\n')
X, tokenizer = tokenize_pad_sequences(df['final_tweet'])
print('After Tokenization & Padding \n', X[0])

y = pd.get_dummies(df.label)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3,
random_state=42, stratify=y_train)

print('Train:         ', X_train.shape, y_train.shape)   
print('Validation Set:', X_valid.shape, y_valid.shape)
print('Test Set:      ', X_test.shape, y_test.shape) 

vocab_size = 5000
embedding_size = 32
epochs = 20

model = Sequential()
model.add(Embedding(vocab_size, embedding_size, input_length= max_len))
model.add(Conv1D(filters=32, kernel_size = 3, padding= 'same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.4))
model.add(Dense(2, activation ='sigmoid'))

plot_model(model, show_shapes = True)

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
print(model.summary())

es = EarlyStopping(monitor = 'val_loss', patience=10)
batch_size = 32

history = model.fit(X_train, y_train,
                    validation_data=(X_valid, y_valid),
                    batch_size=batch_size, epochs=epochs, verbose=1,
                    callbacks = [es])

# Đánh giá mô hình trên tập kiểm tra
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

# Chỉ số bản in
print('Accuracy : {:.4f}'.format(accuracy))

plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)  
plt.plot(history.history['loss'], 'b--', label ='loss')
plt.plot(history.history['val_loss'], 'r:', label ='Val_loss')
plt.xlabel('Epochs')
plt.legend()

plt.subplot(1, 2, 2)  
plt.plot(history.history['accuracy'], 'b--', label ='acc')
plt.plot(history.history['val_accuracy'], 'r:', label ='Val_acc')
plt.xlabel('Epochs')
plt.legend()
plt.show()

# Đồ thị của ma trận nhầm lẫn
y_pred = model.predict(X_test)
plot_cm(np.argmax(np.array(y_test), axis=1), np.argmax(y_pred, axis=1),'LSTM')

# Đổ thị sơ đồ giá trị dự đoán so với sơ đồ giá trị thực
plot_real_pred_val(np.argmax(np.array(y_test), axis=1), np.argmax(y_pred, axis=1),'LSTM')

########################### CNN ###############################################

MAX_FEATURES = 12000
MAX_LENGTH = 1000
def build_model():
    sequences = layers.Input(shape=(MAX_LENGTH,))
    embedded = layers.Embedding(MAX_FEATURES, 32)(sequences)
    x = layers.Conv1D(32, 3, activation='relu')(embedded)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(3)(x)
    x = layers.Conv1D(32, 5, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(5)(x)
    x = layers.Conv1D(32, 5, activation='relu')(x)
    x = layers.GlobalMaxPool1D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation='relu')(x)
    predictions = layers.Dense(2, activation='sigmoid')(x)
    model = models.Model(inputs=sequences, outputs=predictions)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
model = build_model()
print(model.summary())

es = EarlyStopping(monitor = 'var_loss', patience=5)

history = model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=20,
    validation_data=(X_valid, y_valid), callbacks =[es]
    )
# Đánh giá mô hình dựa trên tập kiểm tra
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

# Chỉ số bản in
print('Accuracy   : {:.4f}'.format(accuracy))

plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)  
plt.plot(history.history['loss'], 'b--', label ='loss')
plt.plot(history.history['val_loss'], 'r:', label ='Val_loss')
plt.xlabel('Epochs')
plt.legend()

plt.subplot(1, 2, 2)  
plt.plot(history.history['accuracy'], 'b--', label ='acc')
plt.plot(history.history['val_accuracy'], 'r:', label ='Val_acc')
plt.xlabel('Epochs')
plt.legend()
plt.show()

# Đồ thị của ma trận nhầm lẫn
y_pred = model.predict(X_test)
plot_cm(np.argmax(np.array(y_test), axis=1), np.argmax(y_pred, axis=1),'CNN')

# Đồ thị sơ đồ giá trị dự đoán so với sơ đồ giá trị thực
plot_real_pred_val(np.argmax(np.array(y_test), axis=1), np.argmax(y_pred, axis=1),'CNN')


  
        
        
    
    
    
    
    

     
    
    
      
    
    
    
    
    
    
    
     