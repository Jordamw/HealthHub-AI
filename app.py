import re
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title='HealthHub AI', page_icon='💚', layout='wide')
st.markdown('''<style>.hero{background:#146b58;color:white;padding:2rem;border-radius:14px;text-align:center}.hero h1{color:white;font-size:3rem;margin:0}.hero p{color:white}.stButton button[kind="primary"]{background:#278345}</style>''',unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>HealthHub AI</h1><p>Integrated Healthcare Artificial Intelligence System</p><p>Your Healthcare AI Applications in One Place</p></div>',unsafe_allow_html=True)
st.caption('Educational prototype only. Not for diagnosis, treatment, emergency triage, or real patient information. Do not enter identifiable health data.')

MODULES=['Home','DiaBot','ChronicCare AI','VitalGuard AI','CareFlow AI','MedCost AI','HealthSentiment AI','HealthHero AI']
page=st.sidebar.radio('Navigate',MODULES)

def home():
 st.header('Welcome to HealthHub AI!')
 for n,d in zip(MODULES[1:],['Diabetes risk-factor education','Rule-based chronic disease assessment','Simulated vital-sign monitoring','ER waiting-time simulation','Insurance charge prediction using kNN','Public-domain healthcare text analysis','Health tips and quiz']):
  st.markdown(f'**{n}** — {d}')

def diabot():
 st.header('DiaBot — Type 2 Diabetes Risk Factors')
 sections={'Demographic Risk Factors':{'Age and Type 2 Diabetes Risk':[3],'Race and Ethnicity':[3,4],'Socioeconomic Factors':[4]},'Genetic Risk Factors':{'Family History':[5],'Genetic Risk':[5,6],'Gene and Environment Interaction':[6]},'Behavioral and Lifestyle Risk Factors':{'Obesity and Body Weight':[7,8],'Physical Activity':[8],'Diet and Nutrition':[8,9],'Smoking':[9]},'Metabolic Risk Factors':{'Prediabetes':[10],'Blood Pressure':[10,11],'Other Metabolic Factors':[11]}}
 pdf=Path('diabetesRiskFactors.pdf.pdf')
 if not pdf.exists():
  st.warning('The original DiaBot reference PDF is not included. Upload it below to enable source-grounded answers.')
  upload=st.file_uploader('Diabetes risk-factor reference PDF',type='pdf')
 else: upload=pdf
 def pages(nums):
  if upload is None: return 'Please upload the source PDF first.'
  from pypdf import PdfReader
  reader=PdfReader(upload)
  return '\n\n'.join(f'Page {n}:\n{(reader.pages[n-1].extract_text() or "No readable text")[:1200]}' if n<=len(reader.pages) else f'Page {n} not available' for n in nums)
 category=st.selectbox('Category',list(sections))
 topic=st.selectbox('Topic',list(sections[category]))
 if st.button('Read topic'): st.text(pages(sections[category][topic]))
 q=st.text_input('Ask a diabetes risk-factor question')
 keys={'family history':(['family','genetic','genes','parent'],[5,6]),'body weight':(['weight','obesity','bmi'],[7,8]),'physical activity':(['exercise','physical activity','workout'],[8]),'diet':(['diet','food','nutrition'],[8,9]),'smoking':(['smoking','tobacco'],[9]),'age':(['age','older'],[3]),'prediabetes':(['prediabetes','glucose','a1c'],[10]),'blood pressure':(['blood pressure','hypertension'],[10,11])}
 if st.button('Submit question'):
  matches=[nums for words,nums in keys.values() if any(w in q.lower() for w in words)]
  st.text(pages(sorted(set(n for group in matches for n in group))) if matches else 'Please ask about a listed diabetes risk factor.')

def chronic():
 st.header('ChronicCare AI — Expert System')
 condition=st.selectbox('Condition',['Type 2 Diabetes','Hypertension','Cardiovascular Disease'])
 with st.form('chronic_form'):
  v={}
  if condition=='Type 2 Diabetes':
   v['glucose']=st.number_input('Blood glucose (mg/dL)',min_value=1.,value=110.);v['sleep']=st.number_input('Sleep hours',min_value=0.,max_value=24.,value=8.)
  if condition=='Hypertension':
   v['systolic']=st.number_input('Systolic BP',min_value=1,value=125);v['diastolic']=st.number_input('Diastolic BP',min_value=1,value=80)
  v['medication']=st.selectbox('Prescribed medication taken?',['yes','no']);v['exercise']=st.selectbox('Physical activity?',['yes','no'])
  if condition!='Cardiovascular Disease':v['diet']=st.selectbox('Diet',['healthy','moderate','poor'])
  else:
   v['smoking']=st.selectbox('Smoking?',['no','yes']);v['chest_pain']=st.selectbox('Chest pain?',['no','yes']);v['breath']=st.selectbox('Shortness of breath?',['no','yes'])
  submit=st.form_submit_button('Evaluate')
 if submit:
  score=0;reasons=[];urgent=False
  def add(n,s):
   nonlocal score;score+=n;reasons.append(s)
  if condition=='Type 2 Diabetes':
   if v['glucose']>180 or v['glucose']<70:add(3,'Glucose outside prototype management range')
   if v['sleep']<7:add(1,'Less than 7 hours sleep')
  if condition=='Hypertension':
   if v['systolic']>=140 or v['diastolic']>=90:add(3,'High blood pressure')
   elif v['systolic']>=130 or v['diastolic']>=80:add(2,'Elevated blood pressure')
  if condition=='Cardiovascular Disease':
   if v['chest_pain']=='yes' or v['breath']=='yes':urgent=True;add(5 if v['chest_pain']==v['breath']=='yes' else 3,'Concerning symptom reported')
   if v['smoking']=='yes':add(2,'Smoking')
  if v['medication']=='no':add(2,'Medication not taken')
  if v['exercise']=='no':add(1,'No physical activity')
  if v.get('diet')=='poor':add(2,'Poor diet')
  elif v.get('diet')=='moderate':add(1,'Moderate diet')
  st.metric('Prototype score',score);st.write('Classification:', 'HIGH CONCERN' if score>5 else 'NEEDS ATTENTION' if score>2 else 'LOW CONCERN');st.write('Rules:',reasons or ['No risk rules triggered'])
  if urgent:st.error('Chest pain or breathing difficulty may be an emergency. Seek immediate medical help for new or severe symptoms.')
 st.info('Rule scores are classroom demonstrations, not validated medical risk scores.')

def vital():
 st.header('VitalGuard AI — Simulated Patient Monitoring')
 if st.button('Generate and classify 15 simulated patients'):
  rows=[]
  for i in range(1,16):
   vals={'systolic':np.random.randint(100,160),'diastolic':np.random.randint(60,100),'heart_rate':np.random.randint(60,100),'oxygen':np.random.randint(95,101)}
   kind=np.random.choice(['stable','medium','high'],p=[.5,.3,.2]);k=np.random.choice(list(vals))
   ranges={'medium':{'systolic':(160,180),'diastolic':(100,120),'heart_rate':(100,120),'oxygen':(91,95)},'high':{'systolic':(180,201),'diastolic':(120,131),'heart_rate':(120,141),'oxygen':(85,91)}}
   if kind!='stable':vals[k]=np.random.randint(*ranges[kind][k])
   high=vals['systolic']>=180 or vals['diastolic']>=120 or vals['heart_rate']>=120 or vals['oxygen']<=90
   med=vals['systolic']>=160 or vals['diastolic']>=100 or vals['heart_rate']>=100 or vals['oxygen']<=94
   rows.append({'Patient':f'P{i:03d}',**vals,'Status':'HIGH RISK' if high else 'MEDIUM RISK' if med else 'STABLE'})
  st.session_state.vital_rows=rows
 if 'vital_rows' in st.session_state:
  df=pd.DataFrame(st.session_state.vital_rows);st.dataframe(df,use_container_width=True);st.bar_chart(df['Status'].value_counts());st.download_button('Download simulated results',df.to_csv(index=False),'vitalguard_simulated.csv')

def careflow():
 st.header('CareFlow AI — ER Simulation')
 if 'care_patients' not in st.session_state:st.session_state.care_patients={};st.session_state.care_time=0;st.session_state.care_alerts=set()
 with st.form('checkin'):
  complaint=st.text_input('Complaint (simulation only)');triage=st.selectbox('Staff-entered triage level (1 highest)',[1,2,3,4,5]);check=st.form_submit_button('Check in simulated patient')
 if check:
  if not complaint.strip():st.error('Complaint required')
  else:
   pid=f'P{len(st.session_state.care_patients)+1:03d}';st.session_state.care_patients[pid]={'id':pid,'complaint':complaint,'triage':triage,'arrival':st.session_state.care_time,'wait':0,'status':'Waiting'};st.success(f'{pid} checked in')
 mins=st.number_input('Advance simulation by minutes',min_value=1,value=30)
 if st.button('Advance simulation time'):
  st.session_state.care_time+=int(mins)
  for p in st.session_state.care_patients.values():
   if p['status']=='Waiting':
    p['wait']=st.session_state.care_time-p['arrival']
    for t in (30,60):
     key=(p['id'],t)
     if p['wait']>=t and key not in st.session_state.care_alerts:st.session_state.care_alerts.add(key);st.warning(f'Staff reminder: {p["id"]} reached {t} minutes waiting')
 st.write('Simulation minute:',st.session_state.care_time)
 if st.session_state.care_patients:
  pid=st.selectbox('Patient to update',list(st.session_state.care_patients));status=st.selectbox('New status',['Being Seen','Discharged'])
  if st.button('Update status'):
   p=st.session_state.care_patients[pid]
   if p['status']=='Discharged':st.warning('Already discharged')
   else:p['status']=status;st.success('Status updated')
  df=pd.DataFrame(st.session_state.care_patients.values());st.dataframe(df[['id','complaint','triage','wait','status']],use_container_width=True)
  st.metric('Total reminders',len(st.session_state.care_alerts));st.bar_chart(df.set_index('id')['wait']);st.bar_chart(df['status'].value_counts())
 st.caption('Reminder thresholds 30 and 60 minutes are educational simulation settings only.')

def medcost():
 st.header('MedCost AI — kNN Insurance Charge Prediction')
 from sklearn.model_selection import train_test_split,GridSearchCV
 from sklearn.pipeline import Pipeline
 from sklearn.preprocessing import StandardScaler
 from sklearn.neighbors import KNeighborsRegressor
 from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
 uploaded=st.file_uploader('Upload medical_cost.csv (required)',type='csv')
 if uploaded is not None:
  df=pd.read_csv(uploaded);required=['age','sex','bmi','children','smoker','charges']
  if not set(required).issubset(df.columns):st.error('Missing columns: '+', '.join(set(required)-set(df.columns)));return
  df=df.copy();df['sex_encoded']=df.sex.astype(str).str.lower().map({'female':0,'male':1});df['smoker_encoded']=df.smoker.astype(str).str.lower().map({'no':0,'yes':1})
  feats=['age','sex_encoded','bmi','children','smoker_encoded']
  if df[feats+['charges']].isna().any().any():st.error('Invalid or missing data');return
  if st.button('Train and optimize kNN'):
   Xtr,Xte,ytr,yte=train_test_split(df[feats],df.charges,test_size=.2,random_state=42)
   grid=GridSearchCV(Pipeline([('scale',StandardScaler()),('knn',KNeighborsRegressor())]),{'knn__n_neighbors':list(range(1,31))},cv=5,scoring='neg_mean_squared_error');grid.fit(Xtr,ytr)
   baseline=Pipeline([('scale',StandardScaler()),('knn',KNeighborsRegressor(n_neighbors=3))]);baseline.fit(Xtr,ytr)
   st.session_state.med_model=grid.best_estimator_;st.session_state.med_test=(yte,grid.predict(Xte),baseline.predict(Xte));st.session_state.med_k=grid.best_params_['knn__n_neighbors'];st.session_state.med_cv=pd.DataFrame({'k':range(1,31),'CV MSE':-grid.cv_results_['mean_test_score']});st.success(f'Training complete! Optimized k = {st.session_state.med_k}; {len(Xtr)} training, {len(Xte)} test records.')
 if 'med_model' in st.session_state:
  y,p,b=st.session_state.med_test
  st.dataframe(pd.DataFrame([{'Model':name,'MAE':mean_absolute_error(y,pred),'RMSE':np.sqrt(mean_squared_error(y,pred)),'R²':r2_score(y,pred)} for name,pred in [('Baseline k=3',b),('Optimized kNN',p)]]))
  st.line_chart(st.session_state.med_cv.set_index('k'))
  with st.form('patient_prediction'):
   age=st.number_input('Age',1,120,35);sex=st.selectbox('Sex',['female','male']);bmi=st.number_input('BMI',0.1,100.,28.5);children=st.number_input('Children',0,30,2);smoker=st.selectbox('Smoker',['no','yes']);go=st.form_submit_button('Predict charges')
  if go:
   row=pd.DataFrame([{'age':age,'sex_encoded':int(sex=='male'),'bmi':bmi,'children':children,'smoker_encoded':int(smoker=='yes')}]);st.metric('Estimated charges',f'${st.session_state.med_model.predict(row)[0]:,.2f}')
  if st.button('Process three demonstration patients automatically'):
   batch=pd.DataFrame([{'age':35,'sex_encoded':0,'bmi':28.5,'children':2,'smoker_encoded':0},{'age':50,'sex_encoded':1,'bmi':32.4,'children':3,'smoker_encoded':1},{'age':25,'sex_encoded':0,'bmi':22.1,'children':0,'smoker_encoded':0}]);batch['Predicted charges']=st.session_state.med_model.predict(batch);st.dataframe(batch);st.download_button('Download predictions',batch.to_csv(index=False),'medcost_batch.csv')
 st.caption('Predictions are dataset-dependent estimates, not actual insurance quotes.')

def sentiment():
 st.header('HealthSentiment AI — Comparative Healthcare Sentiment Analysis')
 st.write('Analyze your original two datasets with TextBlob and VADER. The agent cleans text, selects sentiment thresholds, evaluates predictions, and produces comparison graphs.')
 st.caption('Use anonymized or synthetic data only. This educational application is not a clinical decision system.')
 drug=st.file_uploader('Upload Drug Reviews (drugsComTest_raw.csv)',type=['csv'],key='sent_drug')
 patient=st.file_uploader('Upload Patient Feedback (patient_feedback_dataset.xlsx)',type=['xlsx'],key='sent_patient')
 st.info('Both original datasets are required for the full comparison. The drug review CSV must contain review and rating; the patient feedback XLSX must contain Feedback and Sentiment (0/1).')
 if st.button('Run comparative sentiment analysis',disabled=drug is None or patient is None):
  import tempfile,io,contextlib
  from pathlib import Path
  import healthsentiment_engine as engine
  with st.spinner('Analyzing datasets and generating charts. This may take a little while...'):
   try:
    with tempfile.TemporaryDirectory() as tmp:
     folder=Path(tmp)
     paths={'drugsComTest_raw.csv':folder/'drugsComTest_raw.csv','patient_feedback_dataset.xlsx':folder/'patient_feedback_dataset.xlsx'}
     paths['drugsComTest_raw.csv'].write_bytes(drug.getvalue())
     paths['patient_feedback_dataset.xlsx'].write_bytes(patient.getvalue())
     engine.OUTPUT_FOLDER=folder/'results';engine.OUTPUT_FOLDER.mkdir(exist_ok=True)
     agent=engine.SentimentTextblob()
     agent.find_file=lambda filename: paths[filename]
     log=io.StringIO()
     with contextlib.redirect_stdout(log):
      agent.run()
     st.success('HealthSentiment analysis completed!')
     st.subheader('Model comparison')
     st.dataframe(pd.DataFrame(agent.results),hide_index=True,use_container_width=True)
     for img in sorted(engine.OUTPUT_FOLDER.glob('*.png')):
      st.image(str(img),caption=img.stem.replace('_',' ').title(),use_container_width=True)
     with st.expander('Analysis log'):
      st.text(log.getvalue())
     files=sorted(engine.OUTPUT_FOLDER.glob('*.csv'))
     if files:
      with io.BytesIO() as archive:
       import zipfile
       with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in files:z.write(f,arcname=f.name)
       st.download_button('Download CSV results',archive.getvalue(),'healthsentiment_results.zip','application/zip')
   except Exception as exc:
    st.error(f'Analysis could not complete: {exc}')

def hero():
 st.header('HealthHero AI — Health Education and Quiz')
 from healthhero_content import TIPS, QUESTIONS
 tab1,tab2=st.tabs(['Health Tips','Take Quiz'])
 with tab1:
  idx=st.slider('Tip number',1,len(TIPS),1);st.info(TIPS[idx-1]);
  with st.expander('View all tips'):st.write('\n'.join(f'{i}. {t}' for i,t in enumerate(TIPS,1)))
 with tab2:
  with st.form('hero_quiz'):
   answers=[]
   for i,q in enumerate(QUESTIONS,1):answers.append(st.radio(f'{i}. {q["question"]}',q['options'],index=None,key=f'hero_{i}'))
   submit=st.form_submit_button('Grade quiz')
  if submit:
   if any(a is None for a in answers):st.warning('Answer every question first')
   else:
    score=sum(a[0]==q['answer'] for a,q in zip(answers,QUESTIONS));st.metric('Quiz score',f'{score}/{len(QUESTIONS)} ({score/len(QUESTIONS):.0%})')
    for i,(a,q) in enumerate(zip(answers,QUESTIONS),1):st.write(f'{i}. {"Correct" if a[0]==q["answer"] else "Incorrect"}. Correct answer: {q["answer"]}. {q["explanation"]}')

if page=='Home':home()
elif page=='DiaBot':diabot()
elif page=='ChronicCare AI':chronic()
elif page=='VitalGuard AI':vital()
elif page=='CareFlow AI':careflow()
elif page=='MedCost AI':medcost()
elif page=='HealthSentiment AI':sentiment()
elif page=='HealthHero AI':hero()
