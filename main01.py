import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from keras.utils import np_utils, generic_utils
from fuction01 import autoencoder_two_subnetwork_LSTM_fine_tuning_1
import matplotlib.pyplot as plt
from sklearn import metrics

def preprocess_data(X, scaler=None, stand = True):
    if not scaler:
        if stand:
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler
def prepare_data(seperate=False):

    all_data1 = np.load('./dataset/sequence/ZEA_data.npy')
    all_data2 = np.load('./dataset/structure/ZEA_data.npy')
    labels = np.load('./dataset/label/ZEA-label.npy')

    all_data1, scaler1 = preprocess_data(all_data1)
    all_data2, scaler2 = preprocess_data(all_data2)

    return np.array(all_data1), np.array(all_data2),labels

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def GSort(G2scale, G3scale, G4scale, cp2scale, cp3scale, cp4scale):
    G1, G2, G3, cp1, cp2, cp3, Gm, cpm = G2scale, G3scale, G4scale, cp2scale, cp3scale, cp4scale, 0.0, 0.0
    if G1 < G2:
        Gm = G1
        G1 = G2
        G2 = Gm
        cpm = cp1
        cp1 = cp2
        cp2 = cpm
    if G1 < G3:
        Gm = G1
        G1 = G3
        G3 = Gm
        cpm = cp1
        cp1 = cp3
        cp3 = cpm
    if G2 < G3:
        Gm = G2
        G2 = G3
        G3 = Gm
        cpm = cp2
        cp2 = cp3
        cp3 = cpm
    return G1, G2, G3, cp1, cp2, cp3
T = 0.5
def GCalculation(cp):
    return abs(2 * cp - 1)
def GreedyFuzzyDecision(Score2, Score3, Score4):
    NumberSample = len(Score2)
    ScoreE = np.zeros((NumberSample, 2), float)
    for IndexSample in range(NumberSample):
        cp2scale, cp3scale, cp4scale = Score2[IndexSample][1], Score3[IndexSample][1], Score4[IndexSample][1]
        G2scale, G3scale, G4scale = GCalculation(cp2scale), GCalculation(cp3scale), GCalculation(cp4scale)
        G1, G2, G3, cp1, cp2, cp3 = GSort(G2scale, G3scale, G4scale, cp2scale, cp3scale, cp4scale)
        cpE = cp1
        if G1 < T:
            cpE = (cp1 + cp2) / 2
            Gnew = 2 * cpE - 1
            if Gnew < T:
                cpE = (cp1 + cp2 + cp3) / 3
        ScoreE[IndexSample][0] = 1 - cpE
        ScoreE[IndexSample][1] = cpE
    return ScoreE

def Evaluation(y_test, resultslabel):
    TP, FP, TN, FN = 0, 0, 0, 0
    AUC = roc_auc_score(y_test, resultslabel)
    for row1 in range(resultslabel.shape[0]):
        for column1 in range(resultslabel.shape[1]):
            if resultslabel[row1][column1] < 0.5:
                resultslabel[row1][column1] = 0
            else:
                resultslabel[row1][column1] = 1
    for row2 in range(y_test.shape[0]):
        if y_test[row2][0] == 0 and y_test[row2][1] == 1 and y_test[row2][0] == resultslabel[row2][0] and y_test[row2][1] == resultslabel[row2][1]:
            TP = TP + 1
        if y_test[row2][0] == 1 and y_test[row2][1] == 0 and y_test[row2][0] != resultslabel[row2][0] and y_test[row2][1] != resultslabel[row2][1]:
            FP = FP + 1
        if y_test[row2][0] == 1 and y_test[row2][1] == 0 and y_test[row2][0] == resultslabel[row2][0] and y_test[row2][1] == resultslabel[row2][1]:
            TN = TN + 1
        if y_test[row2][0] == 0 and y_test[row2][1] == 1 and y_test[row2][0] != resultslabel[row2][0] and y_test[row2][1] != resultslabel[row2][1]:
            FN = FN + 1
    if TP + FN != 0:
        SEN = TP / (TP + FN)
    else:
        SEN = 999999
    if TN + FP != 0:
        SPE = TN / (TN + FP)
    else:
        SPE = 999999
    if TP + TN + FP + FN != 0:
        ACC = (TP + TN) / (TP + TN + FP + FN)
    else:
        ACC = 999999
    if TP + FP + FN != 0:
        F1 = (2 * TP) / (2 * TP + FP + FN)
    else:
        F1 = 999999
    if TP + FP != 0:
        Pre = TP / (TP + FP)
    else:
        Pre = 999999
    if TP + FP != 0 and TP + FN != 0 and TN + FP != 0 and TN + FN != 0:
        MCC = ((TP * TN) - (TP * FN))/ np.sqrt((TP + FP )*( TP + FN )*( TN + FP )*( TN + FN ))
    else:
        MCC = 999999
    return TP, FP, TN, FN, SEN, SPE, ACC, F1, Pre, MCC, AUC

def DeepInteract():
    X_data1, X_data2 , labels= prepare_data(seperate = True)
    y, encoder = preprocess_labels(labels)# labels labels_new
    

    TPsum, FPsum, TNsum, FNsum, SENsum, SPEsum, ACCsum, F1sum, AUCsum, Presum, MCCsum = [], [], [], [], [], [], [], [], [], [], []
    num_cross_val = 5

    all_prob = {}

    all_prob[0] = []
    all_prob[1] = []
    all_prob[2] = []
    all_prob[3] = []

    for fold in range(num_cross_val):
        train1 = np.array([x for i, x in enumerate(X_data1) if i % num_cross_val != fold])
        test1 = np.array([x for i, x in enumerate(X_data1) if i % num_cross_val == fold])
        train2 = np.array([x for i, x in enumerate(X_data2) if i % num_cross_val != fold])
        test2 = np.array([x for i, x in enumerate(X_data2) if i % num_cross_val == fold])
        train_label = np.array([x for i, x in enumerate(y) if i % num_cross_val != fold])
        test_label = np.array([x for i, x in enumerate(y) if i % num_cross_val == fold])
	
          
        real_labels = []
        for val in test_label:
            if val[0] == 1:
                real_labels.append(0)
            else:
                real_labels.append(1)
	
        train_label_new = []
        for val in train_label:
            if val[0] == 1:
                train_label_new.append(0)
            else:
                train_label_new.append(1)

        prefilter_train, prefilter_test, prefilter_train_bef, prefilter_test_bef, X_train1_tmp_bef, X_test1_tmp_bef, conbine_train,conbine_test = autoencoder_two_subnetwork_LSTM_fine_tuning_1(train1, train2, train_label, test1, test2, test_label)

        clf1 = ExtraTreesClassifier(n_estimators=50)
        clf1.fit(prefilter_train, train_label_new)
        score1 = clf1.predict_proba(prefilter_test)


        clf2 = ExtraTreesClassifier(n_estimators=50)
        clf2.fit(prefilter_train_bef, train_label_new)
        score2 = clf2.predict_proba(prefilter_test_bef)


        clf3 = ExtraTreesClassifier(n_estimators=50)
        clf3.fit(conbine_train, train_label_new)
        score3 = clf3.predict_proba(conbine_test)
        
        
        print('##################### Model 4-2965 completed #####################\n')
        ScoreE = GreedyFuzzyDecision(score1, score2, score3)
        # ScoreE = np.concatenate((score1,score2,score3),axis=1)
        print('##################### Ensemble based on greedy fuzzy decision completed #####################\n')
        TP, FP, TN, FN, SEN, SPE, ACC, F1, Pre, MCC, AUC = Evaluation(test_label,ScoreE)
        tpr, fpr, _ = metrics.roc_curve(test_label, ScoreE)
        print('##################### Evalucation completed #####################\n')

        print('The ' + str(fold + 1) + '-fold cross validation result')
        print('TP:', TP, 'FP:', FP, 'TN:', TN, 'FN:', FN)
        print('TPR:', SEN, 'TNR:', SPE, 'ACC:', ACC, 'F1:', F1, 'Pre:', Pre, 'MCC:', MCC)
    
        TPsum.append(TP)
        FPsum.append(FP)
        TNsum.append(TN)
        FNsum.append(FN)
        SENsum.append(SEN)
        SPEsum.append(SPE)
        ACCsum.append(ACC)
        F1sum.append(F1)
        AUCsum.append(AUC)
        Presum.append(Pre)
        MCCsum.append(MCC)




    # print the average results
    print('The average results')
    print('\ntest mean ACC: ', np.mean(ACCsum))
    print('\ntest mean Pre: ', np.mean(Presum))
    print('\ntest mean Sn: ', np.mean(SENsum))
    print('\ntest mean Sp: ', np.mean(SPEsum))
    print('\ntest mean MCC: ', np.mean(MCCsum))
    print('\ntest mean AUC: ', np.mean(AUCsum))
    


    
    
def transfer_label_from_prob(proba):
    label = [1 if val >= 0.5 else 0 for val in proba]
    return label


if __name__=="__main__":
    DeepInteract()