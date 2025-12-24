# Day 26: 遷移學習 (Transfer Learning) - 站在巨人的肩膀上
# ---------------------------------------------------------
# 昨天我們自己訓練 CNN，準確率約 75%。
# 今天我們要使用 "遷移學習"，直接拿 Google/Microsoft 訓練好的超強模型 (VGG16) 來用。
# VGG16 已經在 ImageNet (1000 萬張圖，1000 類) 上訓練過了，它已經「學會看圖」了。
# 我們只要把它的「腦袋 (卷積層)」借來用，換上我們自己的「眼睛 (分類層)」即可。
# ---------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# --- 1. 準備資料 (Data Preparation) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
pic_dir = os.path.join(SCRIPT_DIR, 'pic')
os.makedirs(pic_dir, exist_ok=True)

print("Loading CIFAR-10 Dataset...")
# 這次我們挑戰難一點的：CIFAR-10 全部 10 類！
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 取樣：為了示範速度，我們只取前 2000 筆訓練資料，500 筆測試資料
# 這更能凸顯遷移學習的威力：資料少也能訓練得很好！
train_size = 2000
test_size = 500
X_train, y_train = X_train[:train_size], y_train[:train_size]
X_test, y_test = X_test[:test_size], y_test[:test_size]

# 正規化 (Normalization)
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot Encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# --- 2. 載入預訓練模型 (Load Pre-trained VGG16) ---
print("\nLoading VGG16 Model...")
# include_top=False: 不要 VGG16 原本的分類層 (因為它是分 1000 類，我們要分 10 類)
# input_shape=(32, 32, 3): 配合 CIFAR-10 的圖片大小
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# 凍結 (Freeze) 卷積層：不讓它們更新權重，因為它們已經學得很好了
for layer in base_model.layers:
    layer.trainable = False

base_model.summary()

# --- 3. 建立新模型 (Build New Model) ---
# 接上我們自己的分類層
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x) # 防止過擬合
predictions = Dense(10, activation='softmax')(x) # 10 類輸出

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\nNew Model Summary:")
# model.summary() # 輸出會很長，先註解掉

# --- 4. 訓練模型 (Training) ---
print("\nTraining Transfer Learning Model...")
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
print("Training complete.")

# --- 5. 模型評估與視覺化 ---
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

# 視覺化 1: 訓練過程
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss History')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy History')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(pic_dir, '26-1_Training_History.png'))
print("Training History plot saved.")

# 視覺化 2: 預測結果展示
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

indices = np.random.choice(len(X_test), 15, replace=False)
images = X_test[indices]
true_labels = np.argmax(y_test[indices], axis=1)
predictions = model.predict(images)
predicted_labels = np.argmax(predictions, axis=1)

plt.figure(figsize=(15, 6))
for i in range(15):
    plt.subplot(3, 5, i + 1)
    plt.imshow(images[i])
    
    color = 'green' if predicted_labels[i] == true_labels[i] else 'red'
    label_text = f"Pred: {class_names[predicted_labels[i]]}\nTrue: {class_names[true_labels[i]]}"
    
    plt.title(label_text, color=color, fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '26-2_Predictions.png'))
print("Predictions plot saved.")

# 視覺化 3: 遷移學習概念圖 (Transfer Learning Concept)
def plot_transfer_learning_concept():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Pre-trained Model (VGG16) - Locked
    rect_vgg = plt.Rectangle((0.1, 0.3), 0.4, 0.4, fc='#FFCC99', ec='black', lw=2)
    ax.add_patch(rect_vgg)
    ax.text(0.3, 0.5, "Pre-trained Model\n(VGG16)\n\nLOCKED 🔒", ha='center', va='center', fontweight='bold')
    ax.text(0.3, 0.25, "Extract Features", ha='center', fontsize=10)

    # Arrow
    ax.arrow(0.5, 0.5, 0.1, 0, head_width=0.05, head_length=0.05, fc='k', ec='k')

    # New Classifier - Trainable
    rect_new = plt.Rectangle((0.65, 0.3), 0.25, 0.4, fc='#99CCFF', ec='black', lw=2)
    ax.add_patch(rect_new)
    ax.text(0.775, 0.5, "New Classifier\n(Dense Layers)\n\nTRAINABLE ✏️", ha='center', va='center', fontweight='bold')
    ax.text(0.775, 0.25, "Classify 10 Classes", ha='center', fontsize=10)
    
    plt.title("Transfer Learning: Don't Reinvent the Wheel", y=0.9)
    plt.savefig(os.path.join(pic_dir, '26-3_Transfer_Learning_Concept.png'))
    print("Transfer Learning Concept plot saved.")

plot_transfer_learning_concept()
