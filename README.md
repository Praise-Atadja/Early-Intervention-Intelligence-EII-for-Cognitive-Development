
# Early-Intervention-Intelligence-EII-for-Cognitive-Development (Autism Prediction)

## Project Overview
This project aims to develop a classification model to predict Autism Spectrum Disorder (ASD) using machine learning techniques. The model will be trained on the Autistic Spectrum Disorder Screening Data for Toddlers dataset, which contains influential features for detecting ASD traits in toddlers. By analyzing behavioral features and individual characteristics from the dataset, we aim to build a robust screening method that can assist healthcare professionals in early autism diagnosis.

## Dataset Used
- **Dataset Name**:Autism Prediction
- **Link to Dataset Source**: [Autism Screening for Toddlers](https://www.kaggle.com/competitions/autismdiagnosis/data)

## Model Training

### Model 1: No Optimization
- **Training Accuracy**: 94.29%
- **Test Accuracy**: 87.92%

### Model 2: No Optimization Regularization L1
- **Training Accuracy**: 80.89%
- **Test Accuracy**: 77.50%

### Model 3: No Optimization Regularization L2
- **Training Accuracy**: 85.71%
- **Test Accuracy**: 90.42%

### Model 4: RMSprop
- **Training Accuracy**: 95.00%
- **Test Accuracy**: 86.67%
- **Optimization Parameters Explanation**: The model is compiled with the RMSprop optimizer, with a learning rate of 0.0011. RMSprop is chosen for its adaptive learning rate capability, which adjusts the learning rate for each parameter individually based on the average of recent gradients for that parameter. This adaptive learning rate can be beneficial in handling sparse gradients and non-stationary objectives, which are common in complex datasets like the one used in this project.

### Model 5: Stochastic Gradient Descent (SGD)
- **Training Accuracy**: 88.75%
- **Test Accuracy**: 85.00%
- **Optimization Parameters Explanation**: The model is compiled with the SGD optimizer, with a learning rate of 0.1 and momentum of 0.9. SGD with momentum is chosen for its ability to accelerate gradient descent in the relevant direction and dampen oscillations. The relatively high learning rate and momentum values are selected to help the model converge faster and navigate through potential local minima more effectively.

### Model 6: Adam Optimization
- **Training Accuracy**: 94.64%
- **Test Accuracy**: 89.17%
- **Optimization Parameters Explanation**: The model is compiled with the Adam optimizer, which uses adaptive moment estimation for computing individual adaptive learning rates for different parameters. Adam combines the advantages of both RMSprop and momentum optimization, making it well-suited for a wide range of optimization problems. In this context, Adam's adaptive learning rate helps in effectively updating the model parameters during training, leading to improved convergence and generalization.

## Findings
- Model 3, which used L2 regularization, achieved the highest test accuracy of 90.42%, indicating that regularization helped prevent overfitting and improved generalization.
- Models using optimization techniques such as RMSprop and Adam also performed well, with test accuracies ranging from 86.67% to 89.17%. These techniques helped in faster convergence during training.
- Model 2, which used L1 regularization, had the lowest test accuracy of 77.50%, indicating that L1 regularization might not have been as effective for this dataset compared to L2 regularization.

## Instructions for Running the Notebook and Loading Saved Models
1. **Dependencies**: Ensure you have the necessary dependencies installed, including Python, Jupyter Notebook, and libraries such as TensorFlow, scikit-learn, pandas, and numpy. You can install dependencies using pip:
   ```
   pip install tensorflow scikit-learn pandas numpy
   ```
2. **Clone the Repository**: Clone this repository to your local machine.
3. **Run the Notebook**: Open the Jupyter Notebook in the repository and execute each cell sequentially. The notebook contains code for data preprocessing, model training, evaluation, and saving models.
4. **Load Saved Models**: Saved models can be loaded using TensorFlow's `load_model` function. Example code:
   ```python
   from tensorflow.keras.models import load_model

   model = load_model('path_to_saved_model')
   ```
   Replace `'path_to_saved_model'` with the actual path to the saved model file.

---

