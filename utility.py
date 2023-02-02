import pandas as pd
import numpy as np
import sklearn
import pickle
import json
import time
import os
import warnings
warnings.filterwarnings("ignore") 


features_names=['EXT_SRC_TOT','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH','DAYS_ID_PUBLISH','DAYS_REGISTRATION','DAYS_LAST_PHONE_CHANGE','HOUR_APPR_PROCESS_START','AMT_INCOME_TOTAL','AMT_ANNUITY','REGION_POPULATION_RELATIVE','NEW_CREDIT_TO_ANNUITY_RATIO','NEW_CREDIT_TO_GOODS_RATIO','AMT_REQ_CREDIT_BUREAU_YEAR','OBS_30_CNT_SOCIAL_CIRCLE','CNT_CHILDREN','AMT_REQ_CREDIT_BUREAU_QRT','FLAG_PHONE','AMT_REQ_CREDIT_BUREAU_MON','WEEKDAY_APPR_PROCESS_START','DEF_30_CNT_SOCIAL_CIRCLE','ORGANIZATION_TYPE','FLAG_OWN_REALTY','FLAG_WORK_PHONE','NAME_FAMILY_STATUS','NAME_TYPE_SUITE','DEF_60_CNT_SOCIAL_CIRCLE','REGION_RATING_CLIENT_W_CITY','REG_CITY_NOT_WORK_CITY','CODE_GENDER']
#########################Encode_Labels_to_Name########################################
Encode_Labels={0:"Non_Defaultor",1:"Defaultor"}

##########################Load_lgbm_Model###################################
with open('./Models/lgbm_pipeline.pkl', 'rb') as f:
    Model = pickle.load(f)

###########################Predict#####################################
def Inference(datapoint):
  data_point_arr=np.array(datapoint)
  x=pd.DataFrame(np.expand_dims(data_point_arr,0),columns=features_names) 
  prob_list=Model.predict_proba(x)[0]
  class_pred=np.argmax(prob_list)
  result="Model classify it as  {}".format(Encode_Labels[class_pred])
  final_result={"Prediction":result,"probability of being defaultor":prob_list[1]  }
  return json.dumps(final_result,ensure_ascii=False)

if __name__=="__main__":

    datapoint=[np.nan,0.6222457752555098,np.nan,-16765,-291,-1186.0,-828.0,11,270000.0,35698.5,0.0035409999999999,36.23408546577587,1.145199203187251,0.0,1.0,0,0.0,1,0.0,'MONDAY',0.0,'School','N',0,'Married','Family',0.0,1,0,'F']
    predictions = Inference(datapoint)
    print(predictions)














    