# HRV

### HRV 데이터로 StressIndex에 어떤 feature가 주요 영향을 미치는 지 확인


##### 모델 생성
```python3

def make_model():
    model = Sequential()
    model.add(Dense(16, input_dim=1, activation='relu'))
    model.add(Dense(32,  activation='relu'))
    model.add(Dense(10,  activation='softmax'))
    
    return model
```
<br>
<br>


##### feature 1개로만 성능 내보기
```python3
for x in X.columns:
    print(x,' train & test')
    X[x] = X[x].values
    X_train, X_test, y_train, y_test = train_test_split(X[x], y, test_size=0.2, random_state=1)
    encoder = LabelEncoder()
    encoder.fit(y)
    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    model = make_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), callbacks=[es, mc])
    loaded_model = load_model('one_model.h5')
    best.append(["{} 정확도 : ".format(x), loaded_model.evaluate(X_test, y_test)[1] ])
```
<br>
 <img width="50%" src="https://user-images.githubusercontent.com/70587454/182335921-e50426f5-dfe2-4caf-8b38-0b9201db54ba.JPG"/>
<br>

##### 수정된 최종 모델
```python3
def model(input_columns):
    model = Sequential([
        Dense(64, input_dim=input_columns, activation='relu'),
        Dense(128),
        Dense(256, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    return model
```

##### 성능이 좋지 않았던 feature은 drop하면서 돌려보기
```python3
score_acc = []
score_loss = []
def kfold_func(kfold,X, y):
    for fold,(train_idx, test_idx) in enumerate(kfold.split(X,y)):
        model2 = model(3) #drop하면서 바꿈 -> 다음부터는 drop도 반복문에 넣어서 처리하기
        model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        encoder2 = LabelEncoder()
        encoder2.fit(y)

        X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
#         X_train, X_test = X.iloc[train_idx], X.iloc[test_idx] #feature 1개만 사용시
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx] 

        y_train = encoder2.transform(y_train)
        y_test = encoder2.transform(y_test)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

    #     es = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='min')
        mc = ModelCheckpoint(filepath='0_model.h5',monitor='val_loss', mode='min', verbose=1, save_best_only=True)

        print('------------------------------------------------------')
        print('Training for {}'.format(fold))
        history = model2.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_test, y_test), callbacks=[mc])
        loaded_model = load_model('0_model.h5')
        score_acc.append(loaded_model.evaluate(X_test, y_test)[1])
        score_loss.append(loaded_model.evaluate(X_test, y_test)[0])  

    print(f'최종 평균 accuracy_score : {sum(score_acc)/len(score_acc)}')
    print(f'최종 평균 loss_score : {sum(score_loss)/len(score_loss)}')
```

<br>
<img width="55%" src="https://user-images.githubusercontent.com/70587454/182336116-f21c1f59-50a3-4ed6-acbf-e0729ca680ff.JPG"/>
<br>
<br>

#### Frequency_domain_features_accuracy
<img width="55%" src="https://user-images.githubusercontent.com/70587454/182335988-ac97a259-c576-4913-a32f-eb3e9c19c582.JPG"/>
<br>

#### 통계_features_accurac
<img width="55%" src="https://user-images.githubusercontent.com/70587454/182336004-8615c360-8917-45ab-a479-f7b6342ad050.JPG"/>
<br>
