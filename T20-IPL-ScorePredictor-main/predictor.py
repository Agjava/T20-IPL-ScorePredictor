### Custom definitions and classes if any ###

def predictRuns(testInput):
    prediction = 0
    ### Your Code Here ###

    import pandas as pd
    import joblib

    with open('lasso.joblib','rb') as f:
        rf = joblib.load(f)
    with open('encoder_bat.joblib','rb') as f:
        encoder = joblib.load(f)
    with open('encoder_bowl.joblib','rb') as f:
        encoder1 = joblib.load(f)
    with open('encoder_team.joblib','rb') as f:
        encoder2 = joblib.load(f)
    with open('encoder_venue.joblib','rb') as f:
        encoder3 = joblib.load(f)

    data = pd.read_csv(testInput)
    dic = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    dic1 = dict(zip(encoder1.classes_, encoder1.transform(encoder1.classes_)))

    #data = pd.get_dummies(data=data, columns=['batting_team', 'bowling_team', 'venue'])
    data['batsmen'] = data['batsmen'].str.split(',')
    #x=len(data['batsmen'])
    data = data.join(pd.DataFrame(data.batsmen.values.tolist(), data.index).add_prefix('batsmen'))
    data['bowlers'] = data['bowlers'].str.split(',')
    data = data.join(pd.DataFrame(data.bowlers.values.tolist(), data.index).add_prefix('bowlers'))

    data['batting_team'] = encoder2.transform(data['batting_team'])
    data['bowling_team'] = encoder2.transform(data['bowling_team'])
    data['venue'] = encoder3.transform(data['venue'])
    #print(data.head())
    
    #dic3 = dict(zip(encoder3.classes_, encoder1.transform(encoder1.classes_)))
    #print(dic3)
    data['batsmen0'] = data['batsmen0'].map(dic) if ('batsmen0' in data.columns) else 0
    data['batsmen1'] = data['batsmen1'].map(dic) if ('batsmen1' in data.columns) else 0
    data['batsmen2'] = data['batsmen2'].map(dic) if ('batsmen2' in data.columns) else 0
    data['batsmen3'] = data['batsmen3'].map(dic) if ('batsmen3' in data.columns) else 0
    data['batsmen4'] = data['batsmen4'].map(dic) if ('batsmen4' in data.columns) else 0
    data['batsmen5'] = data['batsmen5'].map(dic) if ('batsmen5' in data.columns) else 0
    data['batsmen6'] = data['batsmen6'].map(dic) if ('batsmen6' in data.columns) else 0
    data['batsmen7'] = data['batsmen7'].map(dic) if ('batsmen7' in data.columns) else 0
    data['batsmen8'] = data['batsmen8'].map(dic) if ('batsmen8' in data.columns) else 0


    data['bowlers0'] = data['bowlers0'].map(dic1) if ('bowlers0' in data.columns) else 0
    data['bowlers1'] = data['bowlers1'].map(dic1) if ('bowlers1' in data.columns) else 0
    data['bowlers2'] = data['bowlers2'].map(dic1) if ('bowlers2' in data.columns) else 0
    data['bowlers3'] = data['bowlers3'].map(dic1) if ('bowlers3' in data.columns) else 0
    data['bowlers4'] = data['bowlers4'].map(dic1) if ('bowlers4' in data.columns) else 0

    data.drop(['batsmen','bowlers'],axis=1,inplace=True)
    
    data = data.fillna(0)
    for i in data.columns:
        #if data[i].dtype == 'float64':
        data[i]=data[i].astype('int64')
    
    
    data = data.apply(pd.to_numeric, errors='ignore')

    prediction = rf.predict(data)
    prediction = prediction.round(0).astype(int)
    #print(prediction)

    return prediction[0]
