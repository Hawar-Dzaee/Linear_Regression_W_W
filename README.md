# Multivariable Linear Regression Web Application

This web application visualizes a multivariable linear regression model with two features and its corresponding loss landscape. Users can interact with the model by adjusting weights (w1, w2) using widgets, and observe how changes affect the loss function and help reach the global minima.


## Features

* Plot of Multivariable Linear Regression Model: Visualizes how the model fits a given dataset with two features.

* Loss Landscape Plot: Shows the Mean Squared Error (MSE) landscape for different values of weights.

* Interactive Widgets: Adjust weights (w1, w2) in real-time and see the effect on the loss function and regression line.

* Equations Display: Displays the equations used in the plots:
      Linear Regression Equation: Used in the first plot.
      Mean Squared Error (MSE) Equation: Used in the second plot.


## Installation

To run this application, you need to have Python installed on your machine. Follow the steps below to set up the environment:

1. Clone the repository:

    `git clone https://github.com/Hawar-Dzaee/Linear_Regression_W_W.git`


2. Install the required packages:

    `pip install -r requirements.txt`

## Usage

To run the application, use the following command:

  `streamlit run main.py`


This will start the Streamlit server, and you can access the web application by navigating to http://localhost:8501 in your web browser.



## Files

  main.py: Contains the code for the web application.
  LR_W_W.ipynb : a notebook that follows main.py (to some extend)
  requirements.txt: Lists the required Python packages.
  LICENSE 



## Example

Once the application is running, you will see two plots:

1. Linear Regression Plot:

  * Shows the regression plane fitting the dataset with two features.
  * Allows adjustment of weights (w1, w2) using sliders.


2. Loss Landscape Plot:

  * Displays the Mean Squared Error (MSE) landscape.
  * Interactive adjustment of weights to observe how the loss changes



## Requirements 

The application requires the following Python packages:

* Streamlit
* Torch
* Numpy
* Plotly



## License

This project is licensed under the MIT License. See the **LICENSE** file for more details.


Contact
For any questions or suggestions, please open an issue or contact [hawardizayee@gmail.com].



