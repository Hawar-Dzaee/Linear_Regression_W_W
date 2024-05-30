import streamlit as st 
import torch
import numpy as np 
import plotly.graph_objects as go
#-----------------------------------------

# Data & Toy model creation 
# The reason for the toy model to have their on inputs (not using the same as Data creation) is purely for visualizaiton purposes... 
# so that the Toy model doesn't look smaller we parameters is being played with.

lower_bound = -2
upper_bound = 2
sample_size = 5

model_lower_bound = -3
model_upper_bound = 3


# inputs for dataset 

secret_w1 = torch.tensor(0.7)
secret_w2 = torch.tensor(0.3)

X1 = torch.linspace(lower_bound,upper_bound,sample_size)
X2 = torch.linspace(lower_bound,upper_bound,sample_size) # to manipulate the shaprpness of the loss function, play with the scale of the features.
y= secret_w1*X1 + secret_w2*X2

mesh_X1 ,mesh_X2 = torch.meshgrid(X1,X2,indexing='ij') # we mish so that datapoints cover an area and not just a line.
y_mesh = (secret_w1*mesh_X1) + (secret_w2*mesh_X2)



# inputs for model 
X1_model = torch.linspace(model_lower_bound,model_upper_bound,sample_size)
X2_model = torch.linspace(model_lower_bound,model_upper_bound,sample_size) 

mesh_X1_model ,mesh_X2_model = torch.meshgrid(X1_model,X2_model,indexing='ij')    # we mish because the toy model is a plane (dahh)

#-----------------------------------------
# Plot Data & model

def generate_plot(w1,w2):

  # use vector for Scatter3d [Data Generation]
  Datapoints = go.Scatter3d(
      x = mesh_X1.flatten(),
      y = mesh_X2.flatten(),
      z = y_mesh.flatten(),
      mode = 'markers'
  )

  # use matrix for Surface [Toy model]
  plane = go.Surface(
    x = mesh_X1_model,
    y = mesh_X2_model,
    z = (w1 * mesh_X1_model) + (w2 * mesh_X2_model),  
    colorscale = ['rgb(27,158,119)','rgb(27,158,119)'],
    opacity = 0.8,
    showscale = False


  )

  layout = go.Layout(
    scene= dict(
      xaxis = dict(title='X1',range=[-20,20]),
      yaxis = dict(title='X2',range=[-20,20]),
      zaxis = dict(title='y',range=[-50,50])
    )
  )

  figure = go.Figure(data=[Datapoints,plane],layout=layout)
  
  return figure


#---------------------------------------------
# Math for loss landscape 


# setting possible parameter(s) combo
chain_size = 100

W1_matrix,W2_matrix = torch.meshgrid(torch.linspace(-6,6,chain_size),torch.linspace(-6,6,chain_size),indexing='ij')

W1 = W1_matrix.flatten()
W2 = W2_matrix.flatten()



# calculating loss for possible parameter(s) combo 
losses = []

for w1,w2 in zip(W1,W2):
  y_hat = mesh_X1*w1 + mesh_X2*w2
  loss = torch.mean((y - y_hat)**2)
  losses.append(loss)


losses_T = torch.tensor(losses)   # changing to tensor, so we `reshape` it later on, when we need it for `go.Surface`


#-------------------------------------
# Plot Loss function landscape, Global minima, ball

def loss_landscape(w1,w2):

  grid = go.Surface(
    x = W1_matrix,
    y = W2_matrix,
    z = losses_T.reshape(W1_matrix.shape),
    name = "loss functions landscape",
    opacity = 0.4
  )

  Global_minima = go.Scatter3d(
    x = (secret_w1,),
    y = (secret_w2,),
    z = (torch.min(losses_T),),
    mode = 'markers',
    marker = dict(color='yellow',size=10,symbol='diamond'),
    name = 'Global minima'
  )

  ball = go.Scatter3d(
    x = (w1,),
    y = (w2,),
    z = (torch.mean( (y - (w1*X1 + w2*X2))**2),),
    mode = 'markers',
    marker = dict(color='red',size = 7),
    name = 'loss'

  )

  layout = go.Layout(
     scene = dict(
        xaxis = dict(title='w1',range=[-3,3]),
        yaxis = dict(title='w2',range=[-3,3]),
     )
  )

  figure = go.Figure(data = [grid,Global_minima,ball])

  return figure 

#---------------------------------
# streamlit 

st.set_page_config(layout='wide')


st.title("Linear Regression : Two Features, No Bias")
st.write('By : Hawar Dzaee')



with st.sidebar:
    st.subheader("Adjust the parameters to minimize the loss")
    w1_val = st.slider("weight 1:  (w1)", min_value=-4.0, max_value=4.0, step=0.1, value= -3.5)
    w2_val = st.slider("weight 2   (w2)", min_value=-4.0, max_value=4.0, step=0.1, value= -3.2)


container = st.container()

with container:
 
    st.write("")  # Add an empty line to create space

    # Create two columns with different widths
    col1, col2 = st.columns([3,3])

    # Plot figure_1 in the first column
    with col1:
        figure_1 = generate_plot(w1_val, w2_val)
        st.plotly_chart(figure_1, use_container_width=True, aspect_ratio=5.0)  # Change aspect ratio to 1.0
        st.latex(r'''\hat{y} = w1X1 + w2X2''')

    # Plot figure_2 in the second column
    with col2:
        figure_2 = loss_landscape(w1_val, w2_val)
        st.plotly_chart(figure_2, use_container_width=True, aspect_ratio=5.0)
        st.latex(r"""\text{MSE(w1,w2)} = \frac{1}{n} \sum_{i=1}^n (\ y_i- (w1X1 + w2X2) )^2""")