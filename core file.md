**🌿 Plant Disease Prediction using CNN**

## **📌 Project Description**

This project uses a **Convolutional Neural Network (CNN)** to detect and classify plant diseases from leaf images. It helps farmers and researchers identify plant diseases quickly and accurately using deep learning techniques.

---

## **🚀 Features**

* 🌱 Automated plant disease detection  
* 🧠 Deep learning-based CNN model  
* 📊 High accuracy using image classification  
* 🖥️ User-friendly web interface with Streamlit  
* 🔌 Future-ready integration with OpenRouter (Claude Models)  
* 📈 Scalable and customizable architecture

---

## **📂 Project Structure**

Plant-Disease-Prediction/  
│── dataset/  
│── models/  
│   └── plant\_disease\_model.h5  
│── notebooks/  
│   └── plant\_disease\_training.ipynb  
│── src/  
│   ├── train.py  
│   ├── predict.py  
│   ├── preprocess.py  
│   └── openrouter\_integration.py  
│── app.py  
│── requirements.txt  
│── config.py  
│── .env  
│── README.md  
---

## **🛠️ Technologies Used**

* Python 3.9+  
* TensorFlow / Keras  
* OpenCV  
* NumPy & Pandas  
* Matplotlib & Seaborn  
* Streamlit  
* Scikit-learn  
* OpenRouter API (Claude Models)

---

## **📊 Dataset**

Download from Kaggle:  
[https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

After downloading, extract it into the `dataset/` directory.

---

## **⚙️ Installation Guide**

### **1️⃣ Clone the Repository**

git clone https://github.com/your-username/Plant-Disease-Prediction.git  
cd Plant-Disease-Prediction

### **2️⃣ Create Virtual Environment**

python \-m venv venv  
venv\\Scripts\\activate

### **3️⃣ Install Dependencies**

pip install \-r requirements.txt  
---

## **▶️ Training the Model**

python src/train.py

The trained model will be saved in the `models/` directory.

---

## **🔍 Running Predictions**

python src/predict.py \--image test.jpg  
---

## **🌐 Run the Web Application**

streamlit run app.py  
---

## **🤖 OpenRouter (Claude Model) Integration**

### **Create a `.env` File**

OPENROUTER\_API\_KEY=your\_openrouter\_api\_key  
---

## **🧠 Model Architecture**

* Convolutional Layers  
* MaxPooling Layers  
* Dropout for Regularization  
* Fully Connected Dense Layers  
* Softmax Output Layer

---

## **📈 Future Enhancements**

* Transfer Learning with ResNet50 and EfficientNet  
* Mobile App Integration  
* Real-time Disease Detection via Camera  
* Deployment on AWS, Azure, or GCP  
* Explainable AI (Grad-CAM)  
* Integration with IoT Sensors

---

## **👨‍💻 Developed By**

Deep Learning Project Team

---

## **📜 License**

This project is licensed under the MIT License.

---

## **📦 requirements.txt**

tensorflow  
numpy  
pandas  
matplotlib  
seaborn  
opencv-python  
scikit-learn  
streamlit  
pillow  
python-dotenv  
requests  
tqdm  
---

## **⚙️ config.py**

import os

BASE\_DIR \= os.path.dirname(os.path.abspath(\_\_file\_\_))

DATASET\_DIR \= os.path.join(BASE\_DIR, "dataset")  
MODEL\_DIR \= os.path.join(BASE\_DIR, "models")  
MODEL\_PATH \= os.path.join(MODEL\_DIR, "plant\_disease\_model.h5")  
IMAGE\_SIZE \= (224, 224\)  
BATCH\_SIZE \= 32  
EPOCHS \= 10  
---

## **🧹 src/preprocess.py**

import cv2  
import numpy as np  
from config import IMAGE\_SIZE

def preprocess\_image(image\_path):  
   image \= cv2.imread(image\_path)  
   image \= cv2.resize(image, IMAGE\_SIZE)  
   image \= image / 255.0  
   image \= np.expand\_dims(image, axis=0)  
   return image  
---

## **🧠 src/train.py**

import os  
import tensorflow as tf  
from tensorflow.keras.preprocessing.image import ImageDataGenerator  
from tensorflow.keras import layers, models  
from config import DATASET\_DIR, MODEL\_PATH, IMAGE\_SIZE, BATCH\_SIZE, EPOCHS

train\_dir \= os.path.join(DATASET\_DIR, "train")  
valid\_dir \= os.path.join(DATASET\_DIR, "valid")

train\_datagen \= ImageDataGenerator(  
   rescale=1.0 / 255,  
   rotation\_range=20,  
   zoom\_range=0.2,  
   horizontal\_flip=True  
)

valid\_datagen \= ImageDataGenerator(rescale=1.0 / 255\)

train\_generator \= train\_datagen.flow\_from\_directory(  
   train\_dir,  
   target\_size=IMAGE\_SIZE,  
   batch\_size=BATCH\_SIZE,  
   class\_mode="categorical"  
)

valid\_generator \= valid\_datagen.flow\_from\_directory(  
   valid\_dir,  
   target\_size=IMAGE\_SIZE,  
   batch\_size=BATCH\_SIZE,  
   class\_mode="categorical"  
)

num\_classes \= train\_generator.num\_classes

model \= models.Sequential(\[  
   layers.Conv2D(32, (3, 3), activation="relu", input\_shape=(224, 224, 3)),  
   layers.MaxPooling2D(2, 2),

   layers.Conv2D(64, (3, 3), activation="relu"),  
   layers.MaxPooling2D(2, 2),

   layers.Conv2D(128, (3, 3), activation="relu"),  
   layers.MaxPooling2D(2, 2),

   layers.Flatten(),  
   layers.Dense(512, activation="relu"),  
   layers.Dropout(0.5),  
   layers.Dense(num\_classes, activation="softmax")  
\])

model.compile(  
   optimizer="adam",  
   loss="categorical\_crossentropy",  
   metrics=\["accuracy"\]  
)

model.summary()

history \= model.fit(  
   train\_generator,  
   validation\_data=valid\_generator,  
   epochs=EPOCHS  
)

os.makedirs(os.path.dirname(MODEL\_PATH), exist\_ok=True)  
model.save(MODEL\_PATH)

print(f"Model saved at {MODEL\_PATH}")  
---

## **🔍 src/predict.py**

import argparse  
import numpy as np  
import tensorflow as tf  
from preprocess import preprocess\_image  
from config import MODEL\_PATH, DATASET\_DIR  
import os

def load\_class\_names():  
   train\_dir \= os.path.join(DATASET\_DIR, "train")  
   return sorted(os.listdir(train\_dir))

def predict(image\_path):  
   model \= tf.keras.models.load\_model(MODEL\_PATH)  
   image \= preprocess\_image(image\_path)  
   predictions \= model.predict(image)  
   class\_names \= load\_class\_names()  
   predicted\_class \= class\_names\[np.argmax(predictions)\]  
   confidence \= np.max(predictions) \* 100  
   return predicted\_class, confidence

if \_\_name\_\_ \== "\_\_main\_\_":  
   parser \= argparse.ArgumentParser()  
   parser.add\_argument("--image", required=True, help="Path to input image")  
   args \= parser.parse\_args()

   label, confidence \= predict(args.image)  
   print(f"Prediction: {label}")  
   print(f"Confidence: {confidence:.2f}%")  
---

## **🌐 app.py (Streamlit Web App)**

import streamlit as st  
import tensorflow as tf  
import numpy as np  
from PIL import Image  
from config import MODEL\_PATH, IMAGE\_SIZE

st.set\_page\_config(page\_title="Plant Disease Predictor", layout="centered")  
st.title("🌿 Plant Disease Prediction using CNN")

@st.cache\_resource  
def load\_model():  
   return tf.keras.models.load\_model(MODEL\_PATH)

model \= load\_model()

def preprocess\_image(image):  
   image \= image.resize(IMAGE\_SIZE)  
   image \= np.array(image) / 255.0  
   image \= np.expand\_dims(image, axis=0)  
   return image

uploaded\_file \= st.file\_uploader("Upload a leaf image", type=\["jpg", "png", "jpeg"\])

if uploaded\_file is not None:  
   image \= Image.open(uploaded\_file)  
   st.image(image, caption="Uploaded Image", use\_container\_width=True)

   processed\_image \= preprocess\_image(image)  
   predictions \= model.predict(processed\_image)  
   predicted\_class \= np.argmax(predictions)  
   confidence \= np.max(predictions) \* 100

   st.success(f"Prediction: {predicted\_class}")  
   st.info(f"Confidence: {confidence:.2f}%")  
---

## **🤖 src/openrouter\_integration.py**

import os  
import requests  
from dotenv import load\_dotenv

load\_dotenv()

API\_KEY \= os.getenv("OPENROUTER\_API\_KEY")  
URL \= "https://openrouter.ai/api/v1/chat/completions"

def get\_disease\_advice(disease\_name):  
   headers \= {  
       "Authorization": f"Bearer {API\_KEY}",  
       "Content-Type": "application/json"  
   }

   payload \= {  
       "model": "anthropic/claude-3-haiku",  
       "messages": \[  
           {  
               "role": "user",  
               "content": f"Provide treatment and precautions for {disease\_name} in plants."  
           }  
       \]  
   }

   response \= requests.post(URL, headers=headers, json=payload)

   if response.status\_code \== 200:  
       return response.json()\["choices"\]\[0\]\["message"\]\["content"\]  
   return "Unable to fetch advice."  
---

## **▶️ How to Run the Project**

\# Step 1: Activate Environment  
venv\\Scripts\\activate

\# Step 2: Train the Model  
python src/train.py

\# Step 3: Run Streamlit App  
streamlit run app.py  
---

