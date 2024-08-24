import streamlit as st 
import pandas as pd
from sklearn.cluster import DBSCAN,KMeans
import matplotlib.pyplot as plt
import numpy as np

st.title("DBSCAN Visualization")


#___________________________________________Hyperparameters__________________________

sidebar=st.sidebar
with sidebar:
    st.title("HyperParametrs")
    
    a_opt=['KMeans','DBCAN']
    algorithm=st.selectbox("Select Algorithm",options=a_opt,index=1)
    
    st.header("DBCAN Parameters")
    epsilon=st.slider("Select eps radius",value=0.5,min_value=0.1,max_value=5.0,step=0.01)
    min_points=st.slider("No.of Minimum Sample",value=5,min_value=0,max_value=10,step=1)
    
    st.header("KMeans Parameters")
    nc=st.slider("No.of Clusters In Kmeans",value=3,max_value=10,min_value=1)
    
    
    
#___________________________________________Dataset Selection_____________________________

expander_data=st.expander("Data")
with expander_data:
    data_options=['Dart','Face','Chrome_logo','Basic','Spiral','Ring','Lines']
    option_selected=st.selectbox(label='Select Your Data',options=data_options)
    
    if option_selected == 'Basic':
        df_basic=pd.read_csv("datasets\simple5.csv")
        st.scatter_chart(data=df_basic,x='x',y='y')
        df=df_basic
        
    elif option_selected == 'Dart':
         df_dart=pd.read_csv("datasets\dart2.csv")
         st.scatter_chart(data=df_dart,x='x',y='y')
         df=df_dart
         
    elif option_selected == 'Face':
         df_face=pd.read_csv("datasets\joker.csv")
         st.scatter_chart(data=df_face,x='x',y='y')
         df=df_face
         
    elif option_selected == 'Chrome_logo':
         df_chrome=pd.read_csv("datasets\chrome.csv")
         st.scatter_chart(data=df_chrome,x='x',y='y')
         df=df_chrome
         
    elif option_selected == 'Spiral':
         df_spiral=pd.read_csv("datasets\spirals.csv")
         st.scatter_chart(data=df_spiral,x='x',y='y')
         df=df_spiral
         
    elif option_selected == 'Ring':
         df_ring=pd.read_csv("datasets\moothiram.csv")
         st.scatter_chart(data=df_ring,x='x',y='y')
         df=df_ring
    
    elif option_selected == 'Lines':
         df_lines=pd.read_csv("datasets\lines2.csv")
         st.scatter_chart(data=df_lines,x='x',y='y')
         df=df_lines     
    
    else:
        df_basic=pd.read_csv("datasets\simple.csv")
        st.scatter_chart(data=df_basic,x='x',y='y')
        df=df_basic
    
if st.button("Run"):
    data=df.iloc[:,:-1].values
    
    #_________________________________Clustered Datapoints Plot__________________________________
    
    def plotting(cluster,x):
        
        fig,ax=plt.subplots()
        
        unique_clusters = np.unique(cluster)
        output_str = "No. of Unique Clusters: " + ", ".join(str(label) for label in unique_clusters)
        st.markdown(output_str)
       
        expander_cluster=st.expander("Clustered Plot")
        with expander_cluster:
            
            if option_selected == 'Basic':
                plt.scatter(x[:,0],x[:,1],c=cluster,cmap='viridis')
                plt.xlabel("X")
                plt.ylabel("Y")
                st.pyplot(fig)
                
            elif option_selected == 'Dart':
                plt.scatter(x[:,0],x[:,1],c=cluster)
                st.pyplot(fig)
                
            elif option_selected == 'Face':
                plt.scatter(x[:,0],x[:,1],c=cluster)
                st.pyplot(fig)
                
            elif option_selected == 'Chrome_logo':
                plt.scatter(x[:,0],x[:,1],c=cluster)
                st.pyplot(fig)
                
            elif option_selected == 'Spiral':
                plt.scatter(x[:,0],x[:,1],c=cluster)
                st.pyplot(fig)
                
                
            elif option_selected == 'Ring':
                plt.scatter(x[:,0],x[:,1],c=cluster)
                st.pyplot(fig)
                
                
            else:
                plt.scatter(x[:,0],x[:,1],c=cluster)
                st.pyplot(fig)
    
    
     #____________________________________Model Training_________________________________________
    
    if algorithm=='KMeans':
        km=KMeans(n_clusters=nc)
        cl=km.fit_predict(data)
        plotting(cl,data)
    else:
        db=DBSCAN(eps=epsilon,min_samples=min_points)
        cl=db.fit_predict(data)
        plotting(cl,data)
    
   
  
    
        
    
        
    
    
        
        
   



