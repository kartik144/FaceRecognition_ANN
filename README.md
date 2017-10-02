# Face Recognition using Artificial Neural Networks
Artificial Neural Network was implemented to achieve the following goals:
* To detect if the person is wearing sunglasses
* To recognize person from the image
* To recognize the pose of the person (i.e. whether he is looking up, center, left or right)

## Training
32x30 pixel images were used (due to computational constraints) to train the neural network. The list of images used to train are stored in list folder.
For face recognition and sunglasses recognition, only center pose images were used.

## Neural Network Structue

The Neural Network for all three subproblems had  one hidden layer

The node structure for each layer is as follows:

|                       | Input Units   | Hidden Units | Outputs |
| --------------------- | ------------- | ------------ | --------|
| Sunglasses Recognizer | 961 | 4 | 1 |
| Face Recognizer | 961 | 21 | 20 |
| Pose Recognizer | 961 | 7 | 21 |

## Testing
The neural network gave following average accuracy on testing:

|                       | Accuracy | 
| --------------------- | ------ |
| Sunglasses Recognizer | 97.58% | 
| Face Recognizer | 98.75% | 
| Pose Recognizer | 91.84% | 
