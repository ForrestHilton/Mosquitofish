import matplotlib.pyplot as plt
from model import SimpleModel
from functions import model
from graph import ordinary_plot_over_time

# Create a new instance of the model

# Run the model for 20 time steps starting with 10 juveniles and 20 adults
juveniles, adults = model.run(SimpleModel.State(100, 100), 20)

ordinary_plot_over_time(juveniles, adults)
