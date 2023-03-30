import matplotlib.pyplot as plt
from model import SimpleModel
from functions import model

# Create a new instance of the model

# Run the model for 20 time steps starting with 10 juveniles and 20 adults
juveniles, adults = model.run(SimpleModel.State(100, 100), 20)

# Plot the results using Matplotlib
plt.plot(range(20), list(juveniles), label="Juveniles")
plt.plot(range(20), list(adults), label="Adults")
plt.xlabel("Time step")
plt.ylabel("Population")
plt.legend()
plt.show()
