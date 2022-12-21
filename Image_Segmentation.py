#%%
#Import packages
import tensorflow as tf
from tensorflow import keras
from keras import layers,losses,optimizers,callbacks
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy import io
import glob, os
import datetime

#%%
#1. Load the data
#1.1.1. Prepare an empty list for the images and masks
train_inputs = []
train_masks = []


train_f_path = os.path.join(os.getcwd(), 'dataset','train')


#%%
#1.1.2 Load the train_inputs using opencv
train_input_dir = os.path.join(train_f_path,'inputs')
for input_file in os.listdir(train_input_dir):
    inp = cv2.imread(os.path.join(train_input_dir,input_file))
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(inp,(128,128))
    train_inputs.append(inp)

#1.1.3. Load the train_masks
train_mask_dir = os.path.join(train_f_path,'masks')
for mask_file in os.listdir(train_mask_dir):
    mask = cv2.imread(os.path.join(train_mask_dir,mask_file),cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask,(128,128))
    train_masks.append(mask)


#%%
#1.1.4. Convert the lists into numpy array
train_inputs_np = np.array(train_inputs)
train_masks_np = np.array(train_masks)


#%%
#1.1.5. Check some examples
plt.figure(figsize=(10,10))
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.imshow(train_inputs_np[i])
    plt.axis('off')
    
plt.show()

plt.figure(figsize=(10,10))
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.imshow(train_masks_np[i])
    plt.axis('off')

plt.show()
#%%
#2. Data preprocessing
#2.1. Expand the mask dimension
train_masks_np_exp = np.expand_dims(train_masks_np,axis=-1)

#Check the mask output
print(np.unique(train_masks_np_exp[0]))

#%%
#2.2. Convert the mask values into class labels
train_converted_masks = np.round(train_masks_np_exp/255).astype(np.int64)


#Check the mask output
print(np.unique(train_converted_masks[0]))



#%%
#2.3. Normalize inputs pixels value
train_converted_inputs = train_inputs_np / 255.0

sample = train_converted_inputs[0]

#%%
#2.4 Perform train-test split
from sklearn.model_selection import train_test_split

SEED = 12345
x_train,x_test,y_train,y_test = train_test_split(train_converted_inputs,train_converted_masks,test_size=0.2,random_state=SEED)

#%%
#2.5. Convert the numpy arrays into tensor 
x_train_tensor = tf.data.Dataset.from_tensor_slices(x_train)
x_test_tensor = tf.data.Dataset.from_tensor_slices(x_test)
y_train_tensor = tf.data.Dataset.from_tensor_slices(y_train)
y_test_tensor = tf.data.Dataset.from_tensor_slices(y_test)

#%%
#2.6. Combine the images and masks using zip
train_dataset = tf.data.Dataset.zip((x_train_tensor,y_train_tensor))
val_dataset = tf.data.Dataset.zip((x_test_tensor,y_test_tensor))


#%%
#[EXTRA] Create a subclass layer for data augmentation
class Augment(layers.Layer):
    def __init__(self,seed=42):
        super().__init__()
        self.augment_inputs = layers.RandomFlip(mode='horizontal',seed=seed)
        self.augment_labels = layers.RandomFlip(mode='horizontal',seed=seed)
        
    def call(self,inputs,labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs,labels
    
#%%
#2.7. Convert into prefetch dataset
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
TRAIN_SIZE = len(train_dataset)
STEPS_PER_EPOCH = TRAIN_SIZE//BATCH_SIZE

train_batches = (
    train_dataset
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

val_batches = val_dataset.batch(BATCH_SIZE)
train_batches=train_dataset.batch(BATCH_SIZE)
#%%
#3. Visualize some examples
def display(display_list):
    plt.figure(figsize=(15,15))
    title = ['Input Image','True Mask','Predicted Mask']
    
    for i in range(len(display_list)):
        plt.subplot(1,len(display_list),i+1)
        plt.title(title[i])
        plt.imshow(keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
        
    plt.show()
    
#%%
for images, masks in train_batches.take(2):
    sample_image,sample_mask = images[0],masks[0]
    display([sample_image,sample_mask])
    
#%%
#4. Create image segmentation model
#4.1. Use a pretrained model as the feature extraction layers
base_model = keras.applications.MobileNetV2(input_shape=[128,128,3],include_top=False)

#4.2. List down some activation layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
    ]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

#Define the feature extraction model
down_stack = keras.Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False

#Define the upsampling path
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels:int):
    inputs = layers.Input(shape=[128,128,3])
    #Apply functional API to construct U-Net
    #Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    
    #Upsampling and establishing the skip connections(concatenation)
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = layers.Concatenate()
        x = concat([x,skip])
        
    #This is the last layer of the model (output layer)
    last = layers.Conv2DTranspose(
        filters=output_channels,kernel_size=3,strides=2,padding='same') #64x64 --> 128x128
    
    x = last(x)
    
    return keras.Model(inputs=inputs, outputs=x)

#%%
#Make of use of the function to construct the entire U-Net
OUTPUT_CLASSES = 2

model = unet_model(output_channels=OUTPUT_CLASSES)
#Compile the model
loss = losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',loss=loss,metrics=['accuracy'])
keras.utils.plot_model(model, show_shapes=True)

print(model.summary())
#%%
#Create functions to show predictions
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask,axis=-1)
    pred_mask = pred_mask[...,tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None,num=1):
    if dataset:
        for image,mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0],mask[0],create_mask(pred_mask)])
            
    else:
        display([sample_image,sample_mask,create_mask(model.predict(sample_image[tf.newaxis,...]))])


#%%
#Test out the show_prediction function
show_predictions()

#%%
#Create a callback to help display results during model training
class DisplayCallback(callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch+1))

# tensorboard callbacks
log_path=os.path.join(os.getcwd(),'logs',datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb=callbacks.TensorBoard(log_dir=log_path)       
#%%
#5. Model training
#Hyperparameters for the model
EPOCHS = 10
VAL_SUBSPLITS = 5
VALIDATION_STEPS = len(val_dataset)//BATCH_SIZE//VAL_SUBSPLITS

history = model.fit(
    train_batches,
    validation_data=val_batches,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    callbacks=[DisplayCallback(),tb])

#%%
#6. model Evaluation
show_predictions(val_batches,3)

#%%
#7.1 Empty list for test inputs and masks
test_inputs= []
test_masks= []
test_f_path= os.path.join(os.getcwd(), 'dataset','test')

#7.2.1. Load the test images using opencv
test_input_dir = os.path.join(test_f_path,'inputs')
for input_file in os.listdir(test_input_dir):
    inp = cv2.imread(os.path.join(test_input_dir,input_file))
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(inp,(128,128))
    test_inputs.append(inp)

#7.2.2. Load the test masks
test_mask_dir = os.path.join(test_f_path,'masks')
for mask_file in os.listdir(test_mask_dir):
    mask = cv2.imread(os.path.join(test_mask_dir,mask_file),cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask,(128,128))
    test_masks.append(mask)    

# Convert the lists into numpy array
test_inputs_np = np.array(test_inputs)
test_masks_np = np.array(test_masks)

# Expand the mask dimension
test_masks_np_exp = np.expand_dims(test_masks_np,axis=-1)

# Convert the mask values into class labels
test_converted_masks = np.round(test_masks_np_exp/255).astype(np.int64)

#Check the mask output
print(np.unique(test_converted_masks[0]))

# Normalize image pixels value
test_converted_inputs = test_inputs_np / 255.0
sample = test_converted_inputs[0]

# convert to tensor
test_input_tensor=tf.data.Dataset.from_tensor_slices(test_converted_inputs)
test_masks_tensor=tf.data.Dataset.from_tensor_slices(test_converted_masks)

# zip test
test_dataset= tf.data.Dataset.zip((test_input_tensor,test_masks_tensor))

# test batches
test_batches = test_dataset.batch(BATCH_SIZE)

show_predictions(test_batches,3)

#%%
#7. Save the model
save_path=os.path.join(os.getcwd(),'model.h5')
model.save(save_path)


