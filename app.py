import streamlit as st
import pandas as pd
import pickle
pickle_in=open("isolation_forest_model.pkl","rb")
isolation_forest_model=pickle.load(pickle_in)
# Streamlit app
def predict_anomaly(CEACC,CCDEF,FAXAE,FBFFD,EDDAB):
    prediction=isolation_forest_model.predict([[CEACC,CCDEF,FAXAE,FBFFD,EDDAB]])
    print(prediction)
    return prediction

def main():   
    
    st.title('Machine Learning Model Deployment with Streamlit')

    html_temp="""
    <div style="background-color:tomato;padding:10px">
    <h2 style ="color:white;text-algn:center;">Anamoly detection
    </div>
    """
    
    st.markdown(html_temp,unsafe_allow_html=True)
    CEACC=st.number_input("CEACC",0.0)
    CCDEF=st.number_input("CCDEF",0.0)
    FAXAE=st.number_input("FAXAE",0.0)
    FBFFD=st.number_input("FBFFD",0.0)
    EDDAB=st.number_input("EDDAB",0.0)
    result=""

    if st.button("Predict"):
       result=predict_anomaly(CEACC,CCDEF,FAXAE,FBFFD,EDDAB)
    st.success("The output is {}".format(result))

if __name__=="__main__" :
    main() 
