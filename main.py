import matplotlib.pyplot as plt
from model import SimpleModel
import functions as fn
from graph import ordinary_plot_over_time

# Create a new instance of the model

# Run the model for 20 time steps starting with 10 juveniles and 20 adults
#juveniles, adults = fn.model.run(SimpleModel.State(100, 100), 20)

#ordinary_plot_over_time(juveniles, adults)

# Now experiment with different cannibalism models

# Beverton Holt

#juveniles, adults = fn.bev_holt_model.run(SimpleModel.State(100, 100), 20)

#ordinary_plot_over_time(juveniles, adults)

# Allee

juveniles, adults = fn.allee_model.run(SimpleModel.State(100, 100), 20)

ordinary_plot_over_time(juveniles, adults)

# Ricker

juveniles, adults = fn.ricker_model.run(SimpleModel.State(100, 100), 20)

ordinary_plot_over_time(juveniles, adults)

# Linear

#juveniles, adults = fn.linear_model.run(SimpleModel.State(100, 100), 20)

#ordinary_plot_over_time(juveniles, adults)


