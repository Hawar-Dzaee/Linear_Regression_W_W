import streamlit as st 
import torch
import numpy as np 
import plotly.graph_objects as go
#-----------------------------------------

# Dataset & Toy model creation

lower_bound = -2
upper_bound = 2
sample_size = 4




# inputs for dataset 

secret_w1 = torch.tensor(0.4)
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
    z = (w1 * X1_mesh) + (w2 * X2_mesh) ,
    colorscale = ['rgb(27,158,119)','rgb(27,158,119)'],
    opacity = 0.8,
    showscale = False


  )

  layout = go.Layout(
    scene= dict(
       
      xaxis = dict( 
         title = dict(text = 'X1',font=dict(size=18)),
         range=[-10,10],
         zeroline=True
         ),

      yaxis = dict(
         title= dict(text ='X2',font=dict(size=18)),
         range=[-10,10],
         zeroline=True
         ),

      zaxis = dict(
         title = dict(text = 'y',font=dict(size=18)),
         range=[-10,10],
         tickangle=0,
         zeroline=True
         ),

      aspectmode='cube',

      # camera = dict(eye=dict(x= -3.15,y= -1.45,z=0.2))
      camera = dict(
                    up=dict(x=0, y=0, z=1),       # default values
                    center=dict(x=0, y=0, z=0),   # default values 
                    eye=dict(x=1.2, y=0.8, z=0.2)
      ),

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
    opacity = 0.5,
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
        xaxis = dict(title='w1',range=[-6,6]),
        yaxis = dict(title='w2',range=[-6,6]),
        zaxis = dict(title ='loss')
     ),

    legend=dict(
      x=1.3,  # Position the legend to the right
      y=0.9,  # Vertically center the legend
      bgcolor='rgba(255, 255, 255, 0.5)',  # Semi-transparent background
      bordercolor='black',
      borderwidth=1
  )
  )

  figure = go.Figure(data = [grid,Global_minima,ball],layout=layout)

  return figure 

#---------------------------------
# streamlit 

st.set_page_config(layout='wide')


st.title("Linear Regression : Two Features, No Bias")
st.write('By : Hawar Dzaee')


with st.sidebar:
    st.subheader("Adjust the parameters to minimize the loss")
    w1_val = st.slider("weight 1:  ($w_{1}$)", min_value=-3.0, max_value=3.0, step=0.1, value= -0.2)
    w2_val = st.slider("weight 2:  ($w_{2}$)", min_value=-3.0, max_value=3.0, step=0.1, value= -2.2)






container = st.container()

with container:
 
    st.write("")  # Add an empty line to create space

    # Create two columns with different widths
    col1, col2 = st.columns([3,3])

    # Plot figure_1 in the first column
    with col1:
        figure_1 = generate_plot(w1_val, w2_val)
        st.plotly_chart(figure_1, use_container_width=True, aspect_ratio=5.0)  # Change aspect ratio to 1.0
        st.latex(r'''\hat{y} = \color{green}{w_{1}}\color{black}X_{1} \color{black}+ \color{green}{w_{2}}\color{black}X_{2}''')
        st.latex(fr'''\hat{{y}} = {w1_val}X_{{1}} + {w2_val}X_{{2}}''')

  

    # Plot figure_2 in the second column
    with col2:
        figure_2 = loss_landscape(w1_val, w2_val)
        st.plotly_chart(figure_2, use_container_width=True, aspect_ratio=5.0)
        st.latex(r"""\text{MSE}(w_{1},w_{2}) = \frac{1}{n} \sum_{i=1}^n (y_{i} - (w_{1} X_{1} + w_{2} X_{2}))^2""")
        st.latex(fr"""\text{{MSE}}({w1_val:.2f},{w2_val:.2f}) = \frac{{1}}{{n}} \sum_{{i=1}}^n (y_{{i}} - ({w1_val:.2f} X_{{1}} + {w2_val:.2f} X_{{2}}))^2""")



