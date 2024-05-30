import streamlit as st 
import torch
import numpy as np 
import plotly.graph_objects as go
#-----------------------------------------

# Dataset & Toy model creation

lower_bound = -2
upper_bound = 2
sample_size = 5

model_lower_bound = -3
model_upper_bound = 3


# inputs for dataset 

secret_w1 = torch.tensor(1.0)
secret_w2 = torch.tensor(1.0)

x1 = torch.linspace(lower_bound,upper_bound,sample_size)
x2 = torch.linspace(lower_bound,upper_bound,sample_size) # to manipulate the shaprpness of the loss function, play with the scale of the features.


X1_mesh ,X2_mesh = torch.meshgrid(x1,x2,indexing='ij')
X1 = X1_mesh.flatten()
X2 = X2_mesh.flatten()

y = (secret_w1*X1) + (secret_w2*X2)



#-----------------------------------------
# Plot Data & model

def generate_plot(w1,w2):

  # use vector for Scatter3d [Data Generation]
  Datapoints = go.Scatter3d(
      x = X1,
      y = X2,
      z = y,
      mode = 'markers'
  )

  # use matrix for Surface [Toy model]
  plane = go.Surface(
    x = X1_mesh,
    y = X2_mesh,
    z = (w1 * X1_mesh) + (w2 * X1_mesh) ,
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

W1_mesh,W2_mesh  = torch.meshgrid(torch.linspace(-6,6,chain_size),torch.linspace(-6,6,chain_size),indexing='ij')

W1 = W1_mesh.flatten()
W2 = W2_mesh.flatten()



# calculating loss for possible parameter(s) combo 
losses = []

for w1,w2 in zip(W1,W2):
  y_hat = X1*w1 + X2*w2
  loss = torch.mean((y - y_hat)**2)
  losses.append(loss)


losses_T = torch.tensor(losses)   # changing to tensor, so we `reshape` it later on, when we need it for `go.Surface`


#-------------------------------------
# Plot Loss function landscape, Global minima, ball

def loss_landscape(w1,w2):

  grid = go.Surface(
    x = W1_mesh,
    y = W2_mesh,
    z = losses_T.reshape(W1_mesh.shape),
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