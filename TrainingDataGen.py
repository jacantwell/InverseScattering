def DataGen(i):
  """Generates an array of field values along the circumference of a circle of a given radius. 
  
     Inputs:
     
     i(int): Iteration counter
     
     Outputs:
     
     fieled(np.array): Array of real, normalized, field values.
     
     p(np.array): Array of scatterer positions for the corresponding field.
     
     n.b This function can only be called with in the TrainGen(), joblib enabled, function as that is where
     its other constants are defined.
  """
      
    p = np.array(random.sample(range(0, 20), 4))
    P = x.reshape((2,2))

    for k in range(c):

        field[k] = fg.Circle_Field_Generator(r,N,P,1,2)[1]
        field[k] = temp[k].real / np.amax(temp[k].real)

        r += step

    return  field, p


def TrainGen(B,N,C,step):
  """Function that creates batche of training data. Each element in the batch has C channels. Channel 1 contains an array of field values along
     the circumference of a circle of radius r. Each proceeding channel has values for a circle of radius (previous radius) + step.
  
     Inputs:
     
     B(int): Batch size.
     
     N(int): Number of field values evaluated.
     
     C(int): Number of channels.
     
     step(int): The increment the radius of the circle is recorded by.
     
     Outputs:
     
     train_in(np.array): An array containg the batch of field values, i.e the input data.
     
     train_out(np.array): An array containg the batch of coordinate values, i.e the output data."""

    r = 25                 #These constats are required for the DataGen function but are defined "gloablly" as I could not get JobLib
    x = [1,2,3,4]          #to work while they were local.
    temp = np.zeros((C,N))

    data = Parallel(n_jobs=-1, prefer="processes", verbose=6)(  #JobLib ensure all cores of the CPU are used to speed up the data generation.
        delayed(DataGen)(num)
        for num in range(B)
    )

    data = np.array(data,dtype=object)  #The JobLib function outputs a single object so it is then split into two seperate numpy arrays.
    train_in = np.zeros((B,C,N))
    train_out = np.zeros((B,4))

    for i in range(B):

        train_in[i] = data[i,0]
        train_out[i] = data[i,1]

    return train_in, train_out
