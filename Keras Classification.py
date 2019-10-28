
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers, optimizers
from sklearn.metrics import confusion_matrix
from pandas import DataFrame

# Parameters of the generators
imgX, imgY = (150, 150)
batch_size = 20
num_train_rows = 2593 # the number of training instances
num_val_rows = 865 # the number of validation instances
num_test_rows = 865 # the number of test instances
# Parameters of the models
epochs_num = 100
steps_per_epoch = int(num_train_rows/batch_size)
validation_steps = int(num_val_rows/batch_size)

# Generators
# Data folders needed: "train", "validation", "test"
datagen = ImageDataGenerator(rescale=1./255)
                        
train_generator = datagen.flow_from_directory("train",
                                              target_size=(imgX, imgY), 
                                              batch_size=batch_size,
                                              class_mode='categorical')

validation_generator = datagen.flow_from_directory("validation",
                                                    target_size=(imgX, imgY), 
                                                    batch_size=batch_size, 
                                                    class_mode='categorical')

test_generator = datagen.flow_from_directory("test",
                                              target_size=(imgX, imgY),
                                              batch_size=1,
                                              shuffle=False,
                                              class_mode='categorical')

y_true = test_generator.classes


# Functions

# Plots the Validation and Training Losses and Accuracies
def plot_results(history):

    acc= history.history['acc']
    val_acc= history.history['val_acc']
    loss = history.history['loss']
    val_loss= history.history['val_loss']
    
    epochs = range(1, len(acc)+1)
    
    plt.figure()
    plt.plot(epochs, acc, 'r', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')   
    plt.title("Training & Validation Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("model accuracy")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title("Training & Validation Loss")
    plt.xlabel("epoch")
    plt.ylabel("model loss")
    plt.legend()
    plt.show()   
    return True

    
# Build Model 1, return a compiled Keras model
def build_model_1():        
    model = models.Sequential()
    # Layer 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(imgX, imgY, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    # Layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # Layer 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu')) 
    model.add(layers.MaxPooling2D((2, 2)))
    # Layer 4
    model.add(layers.Flatten()) 
    # Layer 5
    model.add(layers.Dense(128, activation='relu'))
    # Layer 6
    model.add(layers.Dense(5, activation='softmax'))
    # Compile
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])       
    return model


# Build Model 2, return a compiled Keras model
def build_model_2(): 
    model = models.Sequential()
    # Layer 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(imgX, imgY, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    # Layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # Layer 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu')) 
    model.add(layers.MaxPooling2D((2, 2)))
    # Layer 4
    model.add(layers.Conv2D(255, (3, 3), activation='relu')) 
    model.add(layers.MaxPooling2D((2, 2)))
    # Layer 5
    model.add(layers.Flatten()) 
    # Layer 6
    model.add(layers.Dropout(0.5))
    # Layer 7
    model.add(layers.Dense(512, activation='relu'))
    # Layer 8
    model.add(layers.Dense(5, activation='softmax'))
    # Compile
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])    
    return model


# Prints and Returns the Loss and the Accuracy of the test set of a Keras model
def evaluate_generator(model):
    print("Evaluating Accuracy and Loss...")
    loss, acc = model.evaluate_generator(test_generator, num_test_rows)   
    print("Validation Loss: ", loss)
    print("Accuracy: ", acc)
    return {'loss':loss, 'acc':acc}


# Returns the predicted probabilities of the test_generator
def get_y_score(model, steps=865):
    print("Evaluating Predicted Probabilities...")
    return model.predict_generator(test_generator, steps=steps)


# Predicts the scores and the classes and returns the classification report
# Assumes an appropriate 'test_generator'
def get_confusion_matrix(model, y_score=None, names = "daisy dandelion rose sunflower tulip".split()):
    if not y_score: 
        y_score = get_y_score(model)
    y_pred = y_score.argmax(1)

    report = confusion_matrix(y_true=y_true, y_pred=y_pred)
    print()
    print(DataFrame(report, columns=names, index=names))
    return report


# Assignment Questions

# Question 1: Build the Network
network1 = build_model_1()


# Question 2: Train the Network
print()
print("NETWORK 1")
history1 = network1.fit_generator(train_generator, 
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs_num, 
                                  validation_data=validation_generator,
                                  validation_steps=validation_steps)

# Account for overfitting
print("""After the chart has been printed estimate where overfitting begins.
Pass the value in the input question""")
plot_results(history1)
overfit_epoch1 = int(input("\n\nOverfit Epoch? "))

# Adjust Model
# Lead the model to the end of epoch before overfitting starts occuring
history1.on_epoch_end(overfit_epoch1) 
model_1 = history1.model
model_1.save("model_1.hf")


# Question 3: Apply the network to the test dataset
evaluate_generator(model_1)

# Print_confusion_matrix
cm_1 = get_confusion_matrix(model_1)


# Question 4: Improve your model
network2 = build_model_2()
print()
print("NETWORK 2")
history2 = network2.fit_generator(train_generator, 
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs_num, 
                                  validation_data=validation_generator,
                                  validation_steps=validation_steps)

# Estimate Overfitting of Improved Model
print("""After the chart has been printed estimate where overfitting begins.
Pass the value in the input question""")
plot_results(history2)
overfit_epoch2 = int(input("\n\nOverfit Epoch? "))

# Adjust Improved Model
# Lead the model on the end of epoch before overfitting starts occuring
history2.on_epoch_end(overfit_epoch2)
model_2 = history2.model
model_2.save("model_2.hf")

# Evaluate Improved Model
evaluate_generator(model_2)

# Print_confusion_matrix
cm_2 = get_confusion_matrix(model_2)


# Question 5: Use data augmentation
# Data Adjustments
datagen_aug = ImageDataGenerator(rotation_range=40, 
                                 width_shift_range=0.2,
                                 height_shift_range=0.2, 
                                 shear_range=0.2, 
                                 zoom_range=0.2,
                                 horizontal_flip=True, 
                                 rescale=1./255,
                                 fill_mode='nearest')

train_generator_aug = datagen_aug.flow_from_directory("train",
                                                      target_size=(imgX, imgY),
                                                      batch_size=batch_size,
                                                      class_mode='categorical')

# Train Model 
# Use the augmented data and fit model as it flows from the directory and use model with best results
print()
print("NETWORK 3")
history3 = build_model_2().fit_generator(train_generator_aug, 
                                         steps_per_epoch=steps_per_epoch,
                                         epochs=epochs_num, 
                                         validation_data=validation_generator,
                                         validation_steps=validation_steps)

# Estimate Overfitting
print("""After the chart has been printed estimate where overfitting begins.
Pass the value in the input question""")
plot_results(history3)
overfit_epoch3 = int(input("\n\nOverfit Epoch? "))

# Adjust Model
# Get the model on the end of epoch before overfitting starts
history3.on_epoch_end(overfit_epoch3) 
model_3 = history3.model
model_3.save("model_3.hf")

# Apply the network to the test dataset
evaluate_generator(model_3)

# Print_confusion_matrix
cm_3 = get_confusion_matrix(model_3)
