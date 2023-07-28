# deep-learning-challenge

<h1>Challenge Report</h1>
<h2>Overview</h2>

<p>The purpose of this challenge is to create a tool that can help a fictional nonprofit foundation (Alphabet Soup) select applicants for funding with the best chance of success in their ventures. We are asked to design a neural network a binary classifier that can predict whether applicants will be successful if funded. </p>
<p>A CSV file has been supplied containing more than 34,000 organisations that have received funding from Alphabet Soup over the years. Within this dataset are ten columns that capture metadata about each organisation. </p>
<p>Using TensorFlow, design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organisation will be successful based on the features in the dataset. Preprocess the data and then compile, train, and evaluate the binary classification model to calculate the model’s loss and accuracy. </p>
<p>Test the model and then experiment with hyperparameter optimisation to improve the model output to 75% or better. 3 additional models are included within he repository showing the results of attempted optimisation, with the final one using </p>

<h3>What is a neural network and how does it work? </h3>

<p>The purpose of a neural network is to approximate complex, non-linear relationships between inputs and outputs in data with the result that they can automatically learn patterns and representations from the data, allowing them to make predictions or decisions based on new, unseen data.</P>
<p> A neural network is composed of interconnected artificial neurons organized into layers. The most common type of neural network is the feedforward neural network, where data flows in one direction from input to output through a series of hidden layers. These consist of: </p><ul>
  <li><b>The input layer:</b> where the raw data is fed into the model.</li>
  <li><b>Hidden Layers:</b> Between the input and output layers, there can be one or more hidden layers. These hidden layers are responsible for learning and transforming the input data through weighted combinations and activation functions.</li>
  <li><b>Forward Propagation:</b> The input data is passed through the neural network layer by layer, with each layer performing a weighted sum and activation function operation. This process continues until the output layer is reached. </b></li>
  <li><b>Weights and Biases:</b> Each connection between neurons in consecutive layers has a weight and a bias associated with it. By adjusting the weights and biases during the training process, the neural network learns to map the inputs to the desired outputs, allowing it to make predictions on new, unseen data with improved accuracy. </li>
  <li><b>Activation Function:</b> After the weighted sum of inputs and biases is computed in each neuron, an activation function is applied to introduce non-linearity to the model. Common activation functions </li>
  <li><b>Loss Function:  </b>The output layer generates predictions. A loss function measures the difference between these predictions and the actual target values. The choice of loss function depends on the task; for example, mean squared error is often used for regression, while cross-entropy loss is used for classification tasks.</li>
  <li><b>Backpropagation:  </b>To train the neural network, the model adjusts its weights and biases based on the calculated loss. Backpropagation is the process of computing the gradients of the loss function with respect to the model's parameters (weights and biases). The optimization algorithm uses this in updating the parameters to reduce loss.</li>
  <li><b>Optimization: </b> Gradient-based optimization algorithms (like Adam) are used to update the weights and biases iteratively, minimizing the loss function.</li>
  <li><b>Iterations (Epochs): </b> The training process consists of multiple iterations (epochs) where the entire dataset is fed through the network. This process continues until the model's performance converges or reaches a satisfactory level.</li></ul>

<h2>Results:</h2>

<h3>Data Preprocessing</h3><ol>
  <li>The data was imported into a pandas’ DataFrame for review and preprocessing.</li>
  
![image](https://github.com/RicT1969/deep-learning-challenge/assets/124494379/1b4df3d4-2b37-4c5f-ace1-6a33c5f4e2c3)

  <li>The EIN and Name columns were dropped as they provided no information relevant to the modelling. </li>
  <li>The number of unique values in each column were extracted and reviewed. The purpose of this was to identify columns with significant numbers of unique categorical values many of which only appear in the dataset on a few occasions. This means that converting the categorical data into numeric data would have produced too large a dataset for efficient processing without meaningful input to the model’s results</li><ul>
  <li>Two columns were identified: </li><ul>
  <li> <b>Application_Type</b> with 17;
  <li> <b>Classification: </b> with 71 unique values. </li></ul>
  <li>For <b>Application _Types</b> values beneath 500 were selected for binning and placed in a category called <b>other</b>; </li>
  <li>Similarly, <b>Classification</b> values under 500 were also chosen for binning also in a category called <b>other</b>. </li></ul>
  <li>Five of the ten columns contained only categorical data. To undertake machine learning, these values are required to be numeric.  The <b>pd.get_dummies</b> method was used to convert the categorical data into binary values, by separating all the unique values into separate columns with a 1 if the value corresponded with the column heading, or 0 if not. </li>
  <li>The column containing the target array (the dependent variable - <b>Is_Succesful</b>) was dropped from the preprocessed data, leaving only the features data (or independent variables) which would form the set of data which would train and test the model. The independent variables were assigned to the variable X, whilst the dependent variable was assigned to the variable y, to be used to train and evaluate the model. </li>
  <li>This meant that the final training data contained 44 columns</li>
  <li>The final step was to scale the data. This is because the ASK_AMT column, containing the funding amount requested within each application, contained a large range of values. This is to achieve normalisation of features – preventing features with larger scales from dominating the learning process, ensuring that all features contribute to the learning process. It also promotes faster convergence on the optimal weights and biases and finally avoiding numerical instability. This is something that can happen when systems such as neural networks are dealing large or small input values. </li><ul>
  <li>The method used Min-Max Scaling (Normalization) which scales the data to a fixed range between 0 and 1. This was achieved using the ‘StandardScaler’ method in sklearn. </li></ul></ol>

<h2>Compiling and training the model</h2>

A deep neural network model was defined using the Keras library with TensorFlow backend. We used a sequential model, which is a linear stack of layers. The defined parameters took the numbers suggested within the starter code, and three layers:<ol>
  <li>First layer</li><ul>
  <li>Nodes: 80 nodes (units). </li>
  <li>The activation function: Rectified Linear Unit (ReLU). </li>
  <li>The input layer: specifying the number of input features in the data. The data used has 44 input features. </li></ul>
  <li>Second layer</li><ul>
  <li>30 nodes (units). </li>
  <li>The activation function: Rectified Linear Unit (ReLU). </li></ul>
  <li>Output Layer: </li><ul></ol>
  <li>Nodes : the output layer of the model with 1 node (unit). Since it's a binary classification problem (based on the "sigmoid" activation), the output node will produce values between 0 and 1, representing the probability of the input belonging to the positive class. </li>
  <li>The activation function:  "sigmoid" squeezes the output between 0 and 1, making it suitable for binary classification problems. </li>
  <li>Epochs: finally, we defined the number of epochs as 100. This is the number of passes through the entire training dataset during the training process. An epoch is completed when the model has seen and processed every sample in the training dataset once. </li>
  <li>The total number of trainable parameters is shown in the summary (the graphic below). These parameters were subsequently updated during the model training process to optimize its performance on the given task. </li></ul>

  ![image](https://github.com/RicT1969/deep-learning-challenge/assets/124494379/6af8ce9b-01ca-4fd8-ac03-ea49cbe24f3b)

  
<h3>Summary of Model 1</h3>

A deep neural network model was compiled using the compile() function in Keras. This used three components. </li><ul>
<li><b>Loss Function (loss): </b> since the model is being used for a binary classification task (output layer with sigmoid activation), the binary cross-entropy loss function was used. This is a common choice for binary classification problems, measuring the difference between the predicted probabilities and the actual binary labels. </li>
<li><b>Optimizer (optimizer): </b> the Adam optimizer is used for training the neural network. Adam stands for Adaptive Moment Estimation, and it is an efficient and widely used optimization algorithm that combines the benefits of both RMSprop and momentum methods. It adapts the learning rates of each parameter during training to improve convergence speed and handle different types of data and architectures effectively. </li>
<li><b>Metrics (metrics): </b> during training, the model's performance will be evaluated based on accuracy. Accuracy is a common metric used for classification tasks, indicating the proportion of correct predictions over the total number of predictions. </li></ul>

<h3>Training</h3><ul>
  
<li>During training, the model aims to minimise the binary cross-entropy loss using the Adam optimizer while evaluating its progress based on accuracy. The training process involves feeding the input data, computing the gradients, and updating the model's parameters (weights and biases) iteratively until convergence or a specified number of epochs is reached. </li></ul>

<h3>Evaluation</h3><ul>
  
<li>The trained deep neural network model was evaluated using the test data (X_test_scaled and y_test). The evaluate() function in Keras was used for this purpose. </li>
<li>The X_test_scaled is the test data containing the input features (scaled) on which the model is evaluated and the y_test is the corresponding ground truth (actual) labels for the test data. </li>
<li>The model evaluation is based on the loss and accuracy of the model compared to the test data. </li></ul>
  
<h3>Results: </h3><ul>


<img width="395" alt="image" src="https://github.com/RicT1969/deep-learning-challenge/assets/124494379/8d64187e-43bb-400a-8ec9-6375c2c359d3">



<li>The model fell short of the aim to get 75% accuracy. </li>
<li>The training data on its final run through got to roughly this figure between 25 and 30 epochs. By the 100th epoch the accuracy was measured at 74.1% and loss at 53.2%, a noticeable difference to the test scores, suggesting that the model was overfitted, and had begun to learn the training data. This is undesirable as it means that the model is less able to generalise to new data sets and is likely to give poor results. </li></ul>

<h2>Opimisation</h2><ul>

<li>A number of models were run with different combinations of hyperparameters. For the purposes of this challenge the hyperparameters chosen were the number of nodes per layer, the number of layers, the number of epochs and different combinations of three activation functions: sigmoid, reLU and tanh. Given this is a binary classification problem, the activation function applied to the output layer remained sigmoid.</li> 
<li>Included here are the results of <b>two</b> of these attempts. The second model includes a significant reduction in epochs, and the third is the most accurate attempt by manual trial and effort.</li>
<li>Finally, keras_tuner was used in order to see whether it was possible to achieve a better result closer or greater than 75%. The auto-tune method represents a “brute “force” method of testing each combination within the range of parameters set. It is expensive in terms of tie, so the range of parameters tested were constrained to the ranges tested in the trial and error attempts, with a maximum of 30 epochs.</li>
<p><b>Note:</b> the results are summarised using the following conventions – the number of nodes are written per layer with a “/” separating the layers; similarly the activation functions used per layer are also seperated by a “/”. The first and last layers refer to the input and output layers respectively.</li></ul>
<h3><b>Summary Model 2</b></h3></ul>
<p><b>Parameters</b></p>
<li>layers: 3</li>
<li>nodes: 25 / 10 / 1</li>
<li>Activation function: relu/relu/sigmoid</li>
<li>Epochs: 30</li></ul>

<img width="344" alt="image" src="https://github.com/RicT1969/deep-learning-challenge/assets/124494379/02a5aa05-ab26-4e3f-af02-71542b21d5f7">


<p><b>Results</b></p><ul>
<li>The model recorded no real difference in terms of accuracy and loss for the testing set</li>
<li>Nonetheless the reduction of epochs to 30 resulted in a closer result between training and testing data: the training data recorded 73.6% accuracy compared to the testing model’s 72.6% showing a closer result and therefore a reduced likelihood of overfitting. </li></ul> 

<img width="439" alt="image" src="https://github.com/RicT1969/deep-learning-challenge/assets/124494379/b5d7a4f0-5c03-4496-954b-59996f517e7e">

<h3><b>Summary Model 3</b></h3>
<p><b>Parameters</b></p>
<li>layers: 4</li>
<li>nodes: 10 / 18 / 8 / 1</li>
<li>Activation function: reLU / tanh / tanh / sigmoid</li>
<li>Epochs: 30</li></ul>

<img width="343" alt="image" src="https://github.com/RicT1969/deep-learning-challenge/assets/124494379/bf189178-56e0-49fa-a92d-9c73a89c1a36">

<p><b>Results</b></p><ul>
<li>The model recorded no real difference in terms of accuracy and loss for the testing set</li>
<li>Nonetheless it achieved a closer result between training and testing data: the training data recorded 73.5% accuracy compared to the testing model’s 72.8% showing a closer result than previously between the two sets. </li>
<li>The results did not show much improvement in accuracy and loss, given the minor variations between the outputs. Broadly the use of four layers, the application of reLU in the first layer and the tanh in the remaining hidden layers appeared the most favourable and gave the best indication that there was no overfitting. </li></ul>

<img width="409" alt="image" src="https://github.com/RicT1969/deep-learning-challenge/assets/124494379/4ecc782d-5731-483c-8ad2-51268a20366a">


<h3><b>Auto-Tune Hyperparameters</b></h3>
<p>Having had disappointing results with manual experimentation, an auto-tuner was used to see if the results could be improved.</p>
<p><b>Parameters</b></p><ul>
<li>layers: 3</li>
<li>nodes for the hidden layer: 26</li>
<li>Activation function for the hidden layer: tanh </li>
<li>Epochs: 20</li></ul>


<p><b>Results</b></p><ul>
<p><i>Report on the parameters for the auto-tune process</i></p>
 <img width="320" alt="image" src="https://github.com/RicT1969/deep-learning-challenge/assets/124494379/cd79b386-867a-4bdb-99d9-26ac1e6e8aaa">
<li>The hyperparameter tuning process has resulted in a neural network model with an accuracy of around 73.21% on the test data.</li>
<li>This is an improvement on our earlier attemptes at hyperparameter tuning, but is still not at the desired minimum 75% accuracy.</li></ul>

<img width="274" alt="image" src="https://github.com/RicT1969/deep-learning-challenge/assets/124494379/73473af4-9657-4304-baae-2eb0c4019630">
<img width="401" alt="image" src="https://github.com/RicT1969/deep-learning-challenge/assets/124494379/3fb29708-a00c-4aa5-91ce-bb60e97cddbe"
  


<p><h3>Further steps</h3></p>
<li>Experiment with different activation functions for the hidden layers trying others like Leaky ReLU, ELU, or SELU.</li>
<li>Implement regularization techniques like L1 and L2 regularization to prevent overfitting. These techniques add penalty terms to the loss function based on the magnitude of weights, encouraging the model to prefer simpler solutions.</li>
<li>Adjust the learning rate of the optimizer to control the step size during gradient updates. A smaller learning rate can make the training process more stable, but it might require more epochs for convergence.</li>
<li>Experiment with different optimizers like RMSprop, Adamax, or Nadam to see if they lead to improved convergence and generalization.</li>
<li>Undertake further feature engineering - for instance Principle Component Analysis will remove features that do not provide much information, simplifying the data set, removing noise and improving the model's efficiency. </li>
<li>Tune the batch size used during training. Smaller batch sizes may introduce more randomness and noise during updates, potentially leading to better generalization.</li>
<li>Experiment with ensemble methods, such as combining the predictions of multiple neural network models or using other machine learning algorithms, like Random Forest or Gradient Boosting, to create a more powerful ensemble model.</li>
<li>Use K-fold cross-validation to get a more reliable estimate of the model's performance and avoid overfitting to a specific train-test split.</li>
<li>Implement early stopping based on the model's performance on the validation set to prevent overfitting and save computation time.</li>

<h2>Sources</h2>
https://stackoverflow.com/questions/24109779/running-get-dummies-on-several-dataframe-columns
https://machinelearningmastery.com/how-to-configure-the-number-of-layers-and-nodes-in-a-neural-network/
https://keras.io/guides/keras_tuner/getting_started/
