
# 필요한 라이브러리 import 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random

import imageio
# import cv2

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from numpy.random import seed
seed(1)
tf.random.set_seed(1)



------------------------------------------------------------------


# artists 데이터 불러오기 및 전처리 


input = "/content/drive/MyDrive/Colab Notebooks/Best_art" # 폴더위치 저장
artists = pd.read_csv(input + '/artists.csv') # artists 정보 csv 불러오기

artists = artists.sort_values(by=['paintings'], ascending=False) # 작품의 개수를 기준으로 화가를 내림차순으로 정렬

artists_top = artists[artists['paintings'] >= 200].reset_index() # 200개 이상의 작품 수를 가진 화가들만 남도록 세팅
artists_top = artists_top[['name', 'paintings']] # 필요없는 칼럼 날리고 이름과 작품 수만 있는 새 dataframe 생성 

artists_top['class_weight'] = artists_top.paintings.sum() / (artists_top.shape[0] * artists_top.paintings) # 화가 별 작품 개수의 차이로 학습에 영향을 줄 수 있으니 각 화가별로 가중치를 계산하여 새 컬럼으로 생성 

class_weights = artists_top['class_weight'].to_dict() # Series를 dictionary로 변환하기



------------------------------------------------------------------


# 화가 작품 이미지 불러오기 및 전처리


images_dir = input + '/images'
artists_dirs = os.listdir(images_dir) # os.listdir을 이용해 디렉토리에 있는 모든 파일 리스트를 가져오기

artists_top_name = artists_top['name'].str.replace(' ', '_').values # 위에서 생성한 artists_top의 name의 화가 이름 띄어쓰기를 _로 변환

# os.path.join으로 인수에 전달된 images_dir, name을 결합하여, 1개의 경로로 만든 후 os.path.exists로 file_path가 실제로 존재하면 경로와 Found를, 존재하지 않으면 경로와 Did not find를 반환
for name in artists_top_name:
    if os.path.exists(os.path.join(images_dir, name)): 
        print("Found -->", os.path.join(images_dir, name))
    else:
        print("Did not find -->", os.path.join(images_dir, name))



------------------------------------------------------------------


# Data Augmentation


batch_size = 8  # Gradient Descent를 한번 계산하기 위한 학습 데이터의 개수
train_input_shape = (224, 224, 3) # (가로 224, 세로 224, RGB 3)
n_classes = artists_top.shape[0] # 종속 변수의 클래스 수


# 학습 데이터의 다양성을 늘리기 위해 데이터 증강(Data augmentation) 수행

# 먼저 ImageDataGenerator 객체 생성. 
#과정: ImageDataGenerator가 원본 데이터 소스 즉, jpg 나 jpeg와 같은 이미지 파일들을 Numpy Array 형태로 가져온 후 사용자가 설정한 여러가지 증강 기법을 적용할 준비를 함. 이 단계를 수행함으로써 ImageDataGenerator 객체가 생성됨
train_datagen = ImageDataGenerator(validation_split=0.2,
                                   rescale=1./255.,
                                   #rotation_range=45,
                                   #width_shift_range=0.5,
                                   #height_shift_range=0.5,
                                   shear_range=5,
                                   #zoom_range=0.7,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                  )

# generator(iterator를 생성해주는 함수, 제너레이터를 만들 때는 일련의 값들을 만들지 않고 이런 값들을 만드는 방법을 만듦)는 훈련용과 검증용으로 두 개 생성
# flow_from_directory를 통해 객체를 numpy array iterator로 생성 및 변환
train_generator = train_datagen.flow_from_directory(directory=images_dir,
                                                    class_mode='categorical', # 화가 10명
                                                    target_size=train_input_shape[0:2],
                                                    batch_size=batch_size,
                                                    subset="training",
                                                    shuffle=True,
                                                    classes=artists_top_name.tolist()
                                                   )

valid_generator = train_datagen.flow_from_directory(directory=images_dir,
                                                    class_mode='categorical',
                                                    target_size=train_input_shape[0:2],
                                                    batch_size=batch_size,
                                                    subset="validation",
                                                    shuffle=True,
                                                    classes=artists_top_name.tolist()
                                                   )


# 최종 배치사이즈 

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size
print("Total number of batches =", STEP_SIZE_TRAIN, "and", STEP_SIZE_VALID) # Total number of batches = 397 and 98



------------------------------------------------------------------


# 원본과 비교하여 증강된 데이터 확인


fig, axes = plt.subplots(1, 2, figsize=(20,10)) 

random_artist = random.choice(artists_top_name)
random_image = random.choice(os.listdir(os.path.join(images_dir, random_artist)))
random_image_file = os.path.join(images_dir, random_artist, random_image)

image = plt.imread(random_image_file)
axes[0].imshow(image)
axes[0].set_title("An original Image of " + random_artist.replace('_', ' '))
axes[0].axis('off')

aug_image = train_datagen.random_transform(image)
axes[1].imshow(aug_image)
axes[1].set_title("A transformed Image of " + random_artist.replace('_', ' '))
axes[1].axis('off')

plt.show()



------------------------------------------------------------------


# Build Model


base_model = MobileNet(weights='imagenet', include_top=False, input_shape=train_input_shape) # pre-trained model인 MobileNet 불러오기 - 전이학습: 밑바닥에서부터 모델을 쌓아올리는 대신에 이미 학습되어있는 패턴들을 활용해서 적용

for layer in base_model.layers:
    layer.trainable = True

# 모델에 layer 추가
X = base_model.output
X = Flatten()(X)

X = Dense(512, kernel_initializer='he_uniform')(X)
X = BatchNormalization()(X)
X = Activation('relu')(X) # 활성화함수(입력된 데이터의 가중 합을 출력 신호로 변환하는 함수)로 relu 사용

X = Dense(16, kernel_initializer='he_uniform')(X)
X = BatchNormalization()(X)
X = Activation('relu')(X)

output = Dense(n_classes, activation='softmax')(X) # 다중 클래스 분류 문제라 softmax

model = Model(inputs=base_model.input, outputs=output)



# 옵티마이저(손실함수를 최소화하는 최적의 가중치를 업데이트 하는 방법)로 아담, 후 컴파일 함수로 환경설정, 에폭수 지정

optimizer = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, 
              metrics=['accuracy']) 

n_epoch = 15

# 콜백함수 정의
early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, 
                           mode='auto', restore_best_weights=True) 

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, 
                              verbose=1, mode='auto') # 모델의 개선이 없을 경우, Learning Rate를 조절해 모델의 개선을 유도



------------------------------------------------------------------


# model train


history1 = model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN,
                              validation_data=valid_generator, validation_steps=STEP_SIZE_VALID,
                              epochs=n_epoch,
                              shuffle=True,
                              verbose=1,
                              callbacks=[reduce_lr],
                              use_multiprocessing=True,
                              workers=16,
                              class_weight=class_weights
                             )

model.save_weights('weight.h5') # weights 저장


score = model.evaluate_generator(train_generator, verbose=1) # 학습모델 평가
print("Prediction accuracy on train data =", score[1])

score = model.evaluate_generator(valid_generator, verbose=1)
print("Prediction accuracy on CV data =", score[1])



# 오차행렬로 성능평가

from sklearn.metrics import *
import seaborn as sns

tick_labels = artists_top_name.tolist()

def showClassficationReport_Generator(model, valid_generator, STEP_SIZE_VALID):
    y_pred, y_true = [], []
    for i in range(STEP_SIZE_VALID):
        (X,y) = next(valid_generator)
        y_pred.append(model.predict(X))
        y_true.append(y)
  
    y_pred = [subresult for result in y_pred for subresult in result]
    y_true = [subresult for result in y_true for subresult in result]
    
    y_true = np.argmax(y_true, axis=1)
    y_true = np.asarray(y_true).ravel()
    
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = np.asarray(y_pred).ravel()
    

    fig, ax = plt.subplots(figsize=(10,10))
    conf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    conf_matrix = conf_matrix/np.sum(conf_matrix, axis=1)
    sns.heatmap(conf_matrix, annot=True, fmt=".2f", square=True, cbar=False, 
                cmap=plt.cm.jet, xticklabels=tick_labels, yticklabels=tick_labels,
                ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix')
    plt.show()
    
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=np.arange(n_classes), target_names=artists_top_name.tolist()))

showClassficationReport_Generator(model, valid_generator, STEP_SIZE_VALID)



# 랜덤 이미지 예측으로 모델 성능 평가

from tensorflow.keras.preprocessing import *

n = 5
fig, axes = plt.subplots(1, n, figsize=(25,10))

for i in range(n):
    random_artist = random.choice(artists_top_name)
    random_image = random.choice(os.listdir(os.path.join(images_dir, random_artist)))
    random_image_file = os.path.join(images_dir, random_artist, random_image)

    test_image = image.load_img(random_image_file, target_size=(train_input_shape[0:2]))

    test_image = image.img_to_array(test_image)
    test_image /= 255.
    test_image = np.expand_dims(test_image, axis=0)

    prediction = model.predict(test_image)
    prediction_probability = np.amax(prediction)
    prediction_idx = np.argmax(prediction)

    labels = train_generator.class_indices
    labels = dict((v,k) for k,v in labels.items())

    title = "Actual artist = {}\nPredicted artist = {}\nPrediction probability = {:.2f} %" \
                .format(random_artist.replace('_', ' '), labels[prediction_idx].replace('_', ' '),
                        prediction_probability*100)

    axes[i].imshow(plt.imread(random_image_file))
    axes[i].set_title(title)
    axes[i].axis('off')

plt.show()



# 실제 웹의 이미지로 예측

url = "https://mblogthumb-phinf.pstatic.net/20140318_57/dohnynose_1395087760818IaqaS_JPEG/T05010_10.jpg?type=w2"

web_image = imageio.imread(url)
# web_image = cv2.resize(web_image, dsize=train_input_shape[0:2], )
web_image = np.resize(web_image, train_input_shape)
web_image = np.array(web_image, dtype='f')
web_image /= 255.
web_image = np.expand_dims(web_image, axis=0)

hund = [100 for i in range(10)] # 예측 결과를 퍼센트로 표현하기 위해 100 곱해줄 list
hund = np.array(hund) 

prediction = model.predict(web_image)
prediction_probability = np.array(prediction)[0]
prediction_probability = np.sort(prediction_probability)[::-1]
prediction_prob = prediction_probability*hund
prediction_idx = np.argsort(prediction)[0][::-1]

prediction_idx_list = []
for i in prediction_idx:
  prediction_idx_list.append(labels[i])

answer_labels = []
prediction_prob_list = []

for i in range(3): # 예측결과 상위 세개만 print 해서 확인
  answer_labels.append(labels[prediction_idx[i]].replace('_', ' '))
  prediction_prob_list.append(prediction_prob[i])
  print(f"{i+1}. Predicted artist = {labels[prediction_idx[i]].replace('_', ' ')}")
  print(f"{i+1}. Prediction probability = {prediction_prob[i]} %")

plt.imshow(imageio.imread(url))
plt.axis('off')
plt.show()

'''
1. Predicted artist = Vincent van Gogh
1. Prediction probability = 86.12021207809448 %
2. Predicted artist = Francisco Goya
2. Prediction probability = 4.340571537613869 %
3. Predicted artist = Rembrandt
3. Prediction probability = 2.394464798271656 %
'''



------------------------------------------------------------------


# 예측결과 상위 3명의 이름과 % 그래프


ans_prob = []

for i in range(3):
  add = str(answer_labels[i]) + '\n' + str(round(prediction_prob_list[i],2))
  ans_prob.append(add)
 
prob = pd.Series(prediction_prob_list)
label_freq = ans_prob

fig = plt.figure(figsize= (10,5))

ax = plt.subplot(1,1,1)

ax = sns.barplot(y=label_freq, x=prob, order= label_freq, alpha=0.6)

ax = plt.xlabel("")
ax = plt.xticks(fontsize= 13, x=-0.03, y=-0.05)
ax = plt.yticks(fontsize= 14, x=-0.01)
ax = plt.title("Label frequency", fontsize= 20, x=0.453, y=1.1)
plt.show()

